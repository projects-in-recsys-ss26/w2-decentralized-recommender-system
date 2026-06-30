import time
import math
import logging
from src.util import *
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import csv

class FedKGClient:
    def __init__(self, config, client_id, client_model):
        """
        Client: Train local model using local dataset and upload local model to server
        FedClient class attributes:
            config: parameters
            model: local model
            result: used to save test results of training local model with local dataset
        """
        self.config = config
        self.id = client_id

        self.model = client_model
        self.GAN = None
        self.result = {'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@5': 0.0, 'HT@10': 0.0, 'HT@20': 0.0}
        dataset = config["dataset"]
        mode = config['data_mode']

        # Choose which trajectory file this client trains on.
        #   merged_client{id}_data.txt -> original data + GAN-synthesised data
        #   client{id}_data.txt        -> original (non-augmented) data only
        # The merged file is produced by the GAN augmentation step in main.py, so
        # we use it when that step is enabled (generate_aug_data) or when the file
        # already exists, and otherwise fall back to the original file. The user
        # can also force the baseline via "use_original_data". This lets training
        # run with and without the GAN module; the training loop below is
        # identical either way -- only the input data changes.
        self.ori_path = f'./data/{mode}/{dataset}/client{self.id}_data.txt'
        merged_path = f'./data/{mode}/{dataset}/merged_client{self.id}_data.txt'
        use_original_data = config.get("use_original_data", "False") == "True"
        gan_augmentation = config.get("generate_aug_data", "False") == "True"

        if use_original_data:
            self.merged_path = self.ori_path
        elif gan_augmentation or os.path.exists(merged_path):
            self.merged_path = merged_path
        else:
            self.merged_path = self.ori_path

        _, self.count = get_local_data(self.ori_path)

    def save_parameters(self, model):
        teacher_model = copy.deepcopy(self.model)
        student_model = copy.deepcopy(model)
        return teacher_model, student_model

    def get_local_dataset(self):
        pass


    def merge_csv(self, original_csv_path, augmented_csv_path, merged_csv_path):
        with open(merged_csv_path, 'w', newline='', encoding='ISO-8859-1') as merged_file:
            csv_writer = csv.writer(merged_file, delimiter='\t')

            # Write original data
            with open(original_csv_path, 'r', encoding='ISO-8859-1') as original_file:
                csv_reader = csv.reader(original_file, delimiter='\t')
                for row in csv_reader:
                    csv_writer.writerow(row)

            # Write augmented data
            with open(augmented_csv_path, 'r', encoding='ISO-8859-1') as augmented_file:
                csv_reader = csv.reader(augmented_file, delimiter='\t')
                for row in csv_reader:
                    csv_writer.writerow(row)

    def pretrain_GAN(self):
        alpha = self.config["alpha"]
        dataset = self.config["dataset"]
        mode = self.config["data_mode"]
        generator_path = f'./gan/{mode}/{dataset}/client_{self.id}/generator_model.pt'
        os.makedirs(os.path.dirname(generator_path), exist_ok=True)
        gan_epochs_list = [50, 3, 10, 3]
        self.GAN.Train_GAN(self.config, self.id, generator_path, gan_epochs_list)


    def _KD_loss(self, pos_logits, neg_logits, soft_pos_logits, soft_neg_logits, T):

        pred_pos = torch.log_softmax(pos_logits / T, dim=1)
        soft_pos = torch.softmax(soft_pos_logits / T, dim=1)

        pred_neg = torch.log_softmax(neg_logits / T, dim=1)
        soft_neg = torch.softmax(soft_neg_logits / T, dim=1)

        loss_pos = torch.nn.functional.kl_div(pred_pos, soft_pos, reduction='batchmean')
        loss_neg = torch.nn.functional.kl_div(pred_neg, soft_neg, reduction='batchmean')

        return (loss_pos + loss_neg) * (T ** 2) / 2

    def kd_train(self, student_model, teacher_model):
        batch_size = 2 * self.config['client_batch_size']
        student_model.train()
        teacher_model.eval()
        personalized_datasets = get_local_data(self.merged_path)
        [user_train, user_valid, user_test, usernum, itemnum], _ = personalized_datasets
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size, self.config['maxlen'], 1, self.config)
        criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.config['lr'], betas=(0.9, 0.98))
        num_batch = len(user_train) // batch_size

        use_softmax = self.config.get("loss_type", "bce") == "sampled_softmax"
        for epoch in range(self.config['kd_epochs']):
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = student_model(u, seq, pos, neg)
                optimizer.zero_grad()
                if use_softmax:
                    sup_loss = sampled_softmax_loss(student_model, seq, pos, self.config.get("num_train_neg", 100), itemnum, self.config['device'])
                else:
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.config['device']), torch.zeros(neg_logits.shape, device=self.config['device'])
                    indices = np.where(pos != 0)
                    sup_loss = criterion(pos_logits[indices], pos_labels[indices])
                    sup_loss += criterion(neg_logits[indices], neg_labels[indices])

                t_pos_logits, t_neg_logits = teacher_model(u, seq, pos, neg)
                loss_kd = self._KD_loss(pos_logits, neg_logits, t_pos_logits, t_neg_logits, self.config['T'])

                loss = self.config["lamda"] * loss_kd + (1 - self.config["lamda"]) * sup_loss # lamda=3
                loss.backward()
                optimizer.step()
        sampler.close()

        return student_model




    def client_update(self):

        config = self.config
        personalized_datasets = get_local_data(self.merged_path)

        [user_train, user_valid, user_test, usernum, itemnum], _ = personalized_datasets

        local_model = deepcopy(self.model)
        sampler = WarpSampler(user_train, usernum, itemnum, config['client_batch_size'], config['maxlen'], 1, config)

        local_model.train()


        criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        # weight_decay adds a second, optimizer-level L2 term on top of the explicit
        # l2_emb penalty below; both default to 0 so behaviour is unchanged unless set.
        optimizer = torch.optim.Adam(
            local_model.parameters(), lr=config['lr'], betas=(0.9, 0.98),
            weight_decay=config.get('weight_decay', 0.0)
        )
        num_batch = len(user_train) // config['client_batch_size']
        if num_batch == 0:
            num_batch = 1  # Ensure at least one batch for training

        # Per-round LR schedule: linear warmup then cosine decay over the local steps.
        # Smooths the otherwise-constant Adam LR and helps every client converge more
        # uniformly. Disabled (constant LR) when use_lr_schedule is not "True".
        total_steps = max(1, config['client_epochs'] * num_batch)
        warmup_steps = max(1, int(config.get('warmup_frac', 0.1) * total_steps))
        use_schedule = config.get('use_lr_schedule', "False") == "True"

        def _lr_factor(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)
            if use_schedule else None
        )

        use_softmax = config.get("loss_type", "bce") == "sampled_softmax"
        for epoch in range(config['client_epochs']):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                optimizer.zero_grad()
                if use_softmax:
                    loss = sampled_softmax_loss(local_model, seq, pos, config.get("num_train_neg", 100), itemnum, config['device'])
                else:
                    pos_logits, neg_logits = local_model(u, seq, pos, neg)
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=config['device']), torch.zeros(neg_logits.shape,device=config['device'])
                    indices = np.where(pos != 0)
                    loss = criterion(pos_logits[indices], pos_labels[indices])
                    loss += criterion(neg_logits[indices], neg_labels[indices])
                for param in local_model.item_emb.parameters(): loss += config['l2_emb'] * torch.norm(param)

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        self.model = deepcopy(local_model)
        sampler.close()

