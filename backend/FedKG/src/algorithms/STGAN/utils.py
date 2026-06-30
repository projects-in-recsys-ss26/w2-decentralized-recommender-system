

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from math import ceil
from torch.autograd import Variable


import numpy as np




def preprocess_data(file_path, max_len, data_time_str):

    def preprocess_coordinates(lat, lon, max_value=5000):
        # Normalize latitude/longitude to [0, 1]
        normalized_lon = (lon + 180) / 360
        normalized_lat = (lat + 90) / 180

        # Map to integers
        int_lon = int(normalized_lon * max_value)
        int_lat = int(normalized_lat * max_value)

        return int_lat, int_lon

    def encode_time(timestamp, data_time_str):
        # Encode the timestamp into year / month / weekday / seconds-of-day
        dt = datetime.strptime(timestamp, data_time_str)
        year = dt.year
        month_of_year = dt.month
        day_of_week = dt.weekday()
        time_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        return year, month_of_year, day_of_week, time_of_day



    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()

    # Initialize the data structure
    user_data = {}

    # Parse every line in the file
    for line in lines:
        parts = line.strip().split('\t')
        user_id, poi_id, class_id, lat, lon, timestamp = int(parts[0]), int(parts[1]), int(parts[2]), float(
            parts[4]), float(parts[5]), parts[-1]


        year, month_of_year, day_of_week, time_of_day = encode_time(timestamp, data_time_str)
        # New user: initialize its data structure
        if user_id not in user_data:
            user_data[user_id] = {
                'user_id': [],
                'poi_id': [],
                'class_id': [],
                'latitude': [],
                'longitude': [],
                'time': []
            }
        # Append the encoded values to that user's data structure
        mapped_lat, mapped_lon = preprocess_coordinates(lat, lon)
        user_data[user_id]['user_id'].append(user_id)
        user_data[user_id]['poi_id'].append(poi_id)
        user_data[user_id]['class_id'].append(class_id)
        user_data[user_id]['latitude'].append(mapped_lat)
        user_data[user_id]['longitude'].append(mapped_lon)
        user_data[user_id]['time'].append(month_of_year)


    # Truncate or pad each user's trajectory to a fixed length
    for user_id, data in user_data.items():
        for key in ['user_id', 'poi_id', 'class_id', 'latitude', 'longitude', 'time']:
            sequence = data[key]
            # Truncate if longer than max_len, pad with zeros if shorter
            if len(sequence) > max_len:
                user_data[user_id][key] = sequence[:max_len]
            else:
                user_data[user_id][key] += [0] * (max_len - len(sequence)) if key != 'time' else [0] * (
                        max_len - len(sequence))

        # Pick CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert the processed data into tensors and move them to the device
    for user_id in user_data.keys():
        user_data[user_id]['user_id'] = torch.tensor(user_data[user_id]['user_id'], dtype=torch.int64).to(device)
        user_data[user_id]['poi_id'] = torch.tensor(user_data[user_id]['poi_id'], dtype=torch.int64).to(device)
        user_data[user_id]['class_id'] = torch.tensor(user_data[user_id]['class_id'], dtype=torch.int64).to(device)
        user_data[user_id]['latitude'] = torch.tensor(user_data[user_id]['latitude'], dtype=torch.float32).to(device)
        user_data[user_id]['longitude'] = torch.tensor(user_data[user_id]['longitude'], dtype=torch.float32).to(device)
        user_data[user_id]['time'] = torch.tensor(user_data[user_id]['time'], dtype=torch.float32).to(device)

    return user_data


def prepare_data_for_discriminator(pos_samples_poi, pos_samples_time, pos_samples_user, pos_samples_lat, pos_samples_lon,
                                   neg_samples_poi, neg_samples_time, neg_samples_user, neg_samples_lat, neg_samples_lon,
                                   gpu=False):
    """
    Build the discriminator's inputs and targets from positive samples (real
    data) and negative samples (generated data).
    """
    device = torch.device("cuda" if gpu else "cpu")

    # Make sure all tensors live on the same device
    pos_samples_poi = pos_samples_poi.to(device)
    pos_samples_time = pos_samples_time.to(device)
    pos_samples_user = pos_samples_user.to(device)
    pos_samples_lat = pos_samples_lat.to(device)
    pos_samples_lon = pos_samples_lon.to(device)

    neg_samples_poi = neg_samples_poi.to(device)
    neg_samples_time = neg_samples_time.to(device)
    neg_samples_user = neg_samples_user.to(device)
    neg_samples_lat = neg_samples_lat.to(device)
    neg_samples_lon = neg_samples_lon.to(device)

    # Concatenate the POI/time/user/lat/lon features of the positive samples
    pos_samples = torch.cat((pos_samples_poi, pos_samples_time, pos_samples_user, pos_samples_lat, pos_samples_lon), dim=1)
    # Concatenate the POI/time/user/lat/lon features of the negative samples
    neg_samples = torch.cat((neg_samples_poi, neg_samples_time, neg_samples_user, neg_samples_lat, neg_samples_lon), dim=1)

    # Stack positive and negative samples together

    inp = torch.cat((pos_samples, neg_samples), dim=0).type(torch.LongTensor).to(device)
    target = torch.zeros(pos_samples.size(0) + neg_samples.size(0), device=device)
    target[:pos_samples.size(0)] = 1  # Label positives as 1, negatives as 0

    # Shuffle
    perm = torch.randperm(inp.size(0))
    inp = inp[perm]
    target = target[perm]

    # Split back into POI/time/user/lat/lon feature blocks
    poi_inp = inp[:, :pos_samples_poi.size(1)]
    time_inp = inp[:, pos_samples_poi.size(1):pos_samples_poi.size(1) + pos_samples_time.size(1)]
    user_inp = inp[:, pos_samples_poi.size(1) + pos_samples_time.size(1):pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1)]
    lat_inp = inp[:, pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1):pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1) + pos_samples_lat.size(1) ]  # latitude block (second to last)
    lon_inp = inp[:, pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1) +  pos_samples_lat.size(1) : pos_samples_poi.size(1) + pos_samples_time.size(1) + pos_samples_user.size(1) +  pos_samples_lat.size(1) + pos_samples_lon.size(1)]  # longitude block (last)


    # Move inputs and target to the correct device
    poi_inp = poi_inp.to(device)
    time_inp = time_inp.to(device)
    user_inp = user_inp.to(device)
    lat_inp = lat_inp.to(device)
    lon_inp = lon_inp.to(device)
    target = target.to(device)

    return (poi_inp, time_inp, user_inp, lat_inp, lon_inp), target

def prepare_batch_data_for_generator(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """
    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target

def sample_from_generator_batch(gen, num_samples, batch_size):
    poi_samples_list = []
    time_samples_list = []
    user_samples_list = []
    lat_samples_list = []  # latitude sample list
    long_samples_list = [] # longitude sample list

    for i in range(int(math.ceil(num_samples / float(batch_size)))):
        poi_samples, time_samples, user_samples, lat_samples, long_samples = gen.sample(batch_size)
        poi_samples_list.append(poi_samples)
        time_samples_list.append(time_samples)
        user_samples_list.append(user_samples)
        lat_samples_list.append(lat_samples)  # add latitude samples
        long_samples_list.append(long_samples) # add longitude samples

    # Concatenate the poi/time/user and lat/lon samples separately
    combined_poi_samples = torch.cat(poi_samples_list, 0)[:num_samples]
    combined_time_samples = torch.cat(time_samples_list, 0)[:num_samples]
    combined_user_samples = torch.cat(user_samples_list, 0)[:num_samples]
    combined_lat_samples = torch.cat(lat_samples_list, 0)[:num_samples]  # concatenate latitude samples
    combined_long_samples = torch.cat(long_samples_list, 0)[:num_samples] # concatenate longitude samples

    return combined_poi_samples, combined_time_samples, combined_user_samples, combined_lat_samples, combined_long_samples

'''def sample_from_generator_batch(gen, num_samples, batch_size):
    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]'''