import pandas as pd
import numpy as np


class DecentralizedRecommender:
    """
    Fully decentralized recommender using Gossip protocol and Local Differential Privacy (LDP).

    Architecture:
        - Each user is an independent node with only their own checkin data.
        - No central server exists at any point.
        - Nodes compute local category-frequency histograms and add Laplace noise (LDP).
        - Nodes exchange and average their noisy histograms via pairwise gossip rounds.
        - After convergence, each node derives its local top-k recommendations.

    Same interface as TimeBasedBaselineRecommender for drop-in evaluation.
    """

    def __init__(self, top_k=5, use_user_clusters=False, epsilon=1.0, gossip_rounds=5):
        """
        Args:
            top_k: Number of top categories to recommend per hour.
            use_user_clusters: Whether to compute cluster-specific recommendations.
            epsilon: LDP privacy budget (lower = more noise = more privacy).
            gossip_rounds: Number of gossip rounds for convergence (more = better convergence).
        """
        self.top_k = top_k
        self.use_user_clusters = use_user_clusters
        self.epsilon = epsilon
        self.gossip_rounds = gossip_rounds

        # Global (without clusters) — same structure as TimeBasedBaselineRecommender
        self.popular_specific_by_hour = {}
        self.global_popular_specific = []
        self.popular_level1_by_hour = {}
        self.global_popular_level1 = []

        # With user clusters (cluster -> hour -> categories)
        self.popular_specific_by_hour_and_cluster = {}
        self.popular_level1_by_hour_and_cluster = {}
        self.global_popular_specific_by_cluster = {}
        self.global_popular_level1_by_cluster = {}

    # =========================================================================
    # Data preparation (identical to centralized version)
    # =========================================================================

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts UTC time to local time and extracts the hour."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
            df['utc_time'] = pd.to_datetime(df['utc_time'])
        df['local_time'] = df['utc_time'] + pd.to_timedelta(df['timezone_offset'], unit='m')
        df['hour'] = df['local_time'].dt.hour
        return df

    # =========================================================================
    # LDP noise mechanism
    # =========================================================================

    def _add_laplace_noise(self, counts_dict: dict) -> dict:
        """Add Laplace noise to a {category: count} dictionary (Local Differential Privacy)."""
        scale = 1.0 / self.epsilon
        noisy = {}
        for cat, count in counts_dict.items():
            noisy_count = count + np.random.laplace(0, scale)
            noisy[cat] = max(0.0, noisy_count)
        return noisy

    # =========================================================================
    # Gossip averaging helpers
    # =========================================================================

    @staticmethod
    def _avg_flat(a: dict, b: dict) -> dict:
        """Average two flat {key: count} dictionaries."""
        keys = set(a) | set(b)
        return {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2.0 for k in keys}

    @classmethod
    def _avg_nested(cls, a: dict, b: dict) -> dict:
        """Average two {outer_key: {inner_key: count}} dictionaries."""
        keys = set(a) | set(b)
        return {k: cls._avg_flat(a.get(k, {}), b.get(k, {})) for k in keys}

    @classmethod
    def _avg_double_nested(cls, a: dict, b: dict) -> dict:
        """Average two {key1: {key2: {key3: count}}} dictionaries."""
        keys = set(a) | set(b)
        return {k: cls._avg_nested(a.get(k, {}), b.get(k, {})) for k in keys}

    # =========================================================================
    # Deep-copy helpers (values are floats/immutable, only dict structure needs copying)
    # =========================================================================

    @staticmethod
    def _copy_flat(d: dict) -> dict:
        return dict(d)

    @staticmethod
    def _copy_nested(d: dict) -> dict:
        return {k: dict(v) for k, v in d.items()}

    @staticmethod
    def _copy_double_nested(d: dict) -> dict:
        return {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in d.items()}

    # =========================================================================
    # Top-K extraction
    # =========================================================================

    def _top_k_from_counts(self, counts_dict: dict) -> list:
        """Extract top_k categories sorted by (noisy) count descending."""
        sorted_cats = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in sorted_cats[:self.top_k]]

    # =========================================================================
    # fit() — Simulates the full decentralized training process
    # =========================================================================

    def fit(self, df: pd.DataFrame, user_cluster_df: pd.DataFrame = None):
        """
        Trains the decentralized model by simulating the Gossip + LDP protocol.

        Steps:
            1. Each user (node) computes local category histograms from their own data only.
            2. Each node adds Laplace noise (LDP) to their histograms before sharing.
            3. Gossip rounds: random pairs of nodes exchange and average their histograms.
            4. After convergence, the final model is extracted from a representative node.

        Args:
            df: DataFrame with checkins (user_id, venue_category_name, level_1, utc_time, timezone_offset)
            user_cluster_df: Optional DataFrame with user_id and cluster columns
        """
        print(f"🌐 Training Decentralized Model (Gossip + LDP, ε={self.epsilon}, rounds={self.gossip_rounds})...")
        df = self._prepare_data(df)

        has_clusters = False
        if user_cluster_df is not None and self.use_user_clusters:
            df = df.merge(user_cluster_df[['user_id', 'cluster']], on='user_id', how='left')
            has_clusters = True

        # =====================================================================
        # PHASE 1: Local computation — each user is an independent node
        # =====================================================================
        user_ids = df['user_id'].unique()
        n_users = len(user_ids)
        print(f"  📱 Phase 1: Computing local histograms for {n_users} nodes...")

        nodes = []
        for uid in user_ids:
            user_data = df[df['user_id'] == uid]
            node = {}

            # --- Global category counts (specific + level_1) with LDP noise ---
            spec_counts = user_data['venue_category_name'].value_counts().to_dict()
            node['global_specific'] = self._add_laplace_noise(spec_counts)

            lvl1_counts = user_data['level_1'].value_counts().to_dict()
            node['global_level1'] = self._add_laplace_noise(lvl1_counts)

            # --- Hourly category counts with LDP noise ---
            node['hourly_specific'] = {}
            node['hourly_level1'] = {}
            for hour, hour_group in user_data.groupby('hour'):
                h_spec = hour_group['venue_category_name'].value_counts().to_dict()
                node['hourly_specific'][hour] = self._add_laplace_noise(h_spec)

                h_lvl1 = hour_group['level_1'].value_counts().to_dict()
                node['hourly_level1'][hour] = self._add_laplace_noise(h_lvl1)

            # --- Cluster-specific counts with LDP noise ---
            if has_clusters:
                node['cluster_global_specific'] = {}
                node['cluster_global_level1'] = {}
                node['cluster_hourly_specific'] = {}
                node['cluster_hourly_level1'] = {}

                # Each user belongs to exactly one cluster — only that cluster gets non-zero counts.
                # Other clusters remain empty (zeros). During gossip averaging, this naturally
                # converges: within-cluster rankings are preserved because the dilution factor
                # from out-of-cluster nodes is identical for all categories.
                if pd.notna(user_data['cluster'].iloc[0]):
                    user_cluster = int(user_data['cluster'].iloc[0])

                    node['cluster_global_specific'][user_cluster] = self._add_laplace_noise(spec_counts)
                    node['cluster_global_level1'][user_cluster] = self._add_laplace_noise(lvl1_counts)

                    node['cluster_hourly_specific'][user_cluster] = {}
                    node['cluster_hourly_level1'][user_cluster] = {}
                    for hour, hour_group in user_data.groupby('hour'):
                        h_spec = hour_group['venue_category_name'].value_counts().to_dict()
                        node['cluster_hourly_specific'][user_cluster][hour] = self._add_laplace_noise(h_spec)

                        h_lvl1 = hour_group['level_1'].value_counts().to_dict()
                        node['cluster_hourly_level1'][user_cluster][hour] = self._add_laplace_noise(h_lvl1)

            nodes.append(node)

        # =====================================================================
        # PHASE 2: Gossip rounds — pairwise averaging, no central server
        # =====================================================================
        print(f"  🔄 Phase 2: Running {self.gossip_rounds} gossip rounds ({n_users} nodes)...")

        for r in range(self.gossip_rounds):
            # Randomly shuffle nodes and form pairs
            indices = np.random.permutation(n_users)
            n_pairs = n_users // 2

            for p in range(n_pairs):
                i, j = indices[2 * p], indices[2 * p + 1]
                node_a = nodes[i]
                node_b = nodes[j]

                # Average global histograms
                avg_g_spec = self._avg_flat(node_a['global_specific'], node_b['global_specific'])
                avg_g_lvl1 = self._avg_flat(node_a['global_level1'], node_b['global_level1'])

                # Average hourly histograms
                avg_h_spec = self._avg_nested(node_a['hourly_specific'], node_b['hourly_specific'])
                avg_h_lvl1 = self._avg_nested(node_a['hourly_level1'], node_b['hourly_level1'])

                # Assign to both nodes (independent copies)
                nodes[i]['global_specific'] = avg_g_spec
                nodes[j]['global_specific'] = self._copy_flat(avg_g_spec)
                nodes[i]['global_level1'] = avg_g_lvl1
                nodes[j]['global_level1'] = self._copy_flat(avg_g_lvl1)

                nodes[i]['hourly_specific'] = avg_h_spec
                nodes[j]['hourly_specific'] = self._copy_nested(avg_h_spec)
                nodes[i]['hourly_level1'] = avg_h_lvl1
                nodes[j]['hourly_level1'] = self._copy_nested(avg_h_lvl1)

                # Average cluster-specific histograms
                if has_clusters:
                    avg_cl_g_spec = self._avg_nested(node_a['cluster_global_specific'], node_b['cluster_global_specific'])
                    avg_cl_g_lvl1 = self._avg_nested(node_a['cluster_global_level1'], node_b['cluster_global_level1'])
                    avg_cl_h_spec = self._avg_double_nested(node_a['cluster_hourly_specific'], node_b['cluster_hourly_specific'])
                    avg_cl_h_lvl1 = self._avg_double_nested(node_a['cluster_hourly_level1'], node_b['cluster_hourly_level1'])

                    nodes[i]['cluster_global_specific'] = avg_cl_g_spec
                    nodes[j]['cluster_global_specific'] = self._copy_nested(avg_cl_g_spec)
                    nodes[i]['cluster_global_level1'] = avg_cl_g_lvl1
                    nodes[j]['cluster_global_level1'] = self._copy_nested(avg_cl_g_lvl1)
                    nodes[i]['cluster_hourly_specific'] = avg_cl_h_spec
                    nodes[j]['cluster_hourly_specific'] = self._copy_double_nested(avg_cl_h_spec)
                    nodes[i]['cluster_hourly_level1'] = avg_cl_h_lvl1
                    nodes[j]['cluster_hourly_level1'] = self._copy_double_nested(avg_cl_h_lvl1)

            print(f"    Round {r + 1}/{self.gossip_rounds} completed.")

        # =====================================================================
        # PHASE 3: Extract final model from converged node
        # =====================================================================
        print("  📊 Phase 3: Extracting final model from converged node...")
        rep = nodes[0]  # After gossip convergence, all nodes have approximately the same histograms

        # Global recommendations
        self.global_popular_specific = self._top_k_from_counts(rep['global_specific'])
        self.global_popular_level1 = self._top_k_from_counts(rep['global_level1'])

        # Hourly recommendations
        for hour, counts in rep['hourly_specific'].items():
            self.popular_specific_by_hour[hour] = self._top_k_from_counts(counts)
        for hour, counts in rep['hourly_level1'].items():
            self.popular_level1_by_hour[hour] = self._top_k_from_counts(counts)

        # Cluster-specific recommendations
        if has_clusters:
            for cluster, counts in rep['cluster_global_specific'].items():
                self.global_popular_specific_by_cluster[cluster] = self._top_k_from_counts(counts)
            for cluster, counts in rep['cluster_global_level1'].items():
                self.global_popular_level1_by_cluster[cluster] = self._top_k_from_counts(counts)

            for cluster, hour_dict in rep['cluster_hourly_specific'].items():
                self.popular_specific_by_hour_and_cluster[cluster] = {}
                for hour, counts in hour_dict.items():
                    self.popular_specific_by_hour_and_cluster[cluster][hour] = self._top_k_from_counts(counts)

            for cluster, hour_dict in rep['cluster_hourly_level1'].items():
                self.popular_level1_by_hour_and_cluster[cluster] = {}
                for hour, counts in hour_dict.items():
                    self.popular_level1_by_hour_and_cluster[cluster][hour] = self._top_k_from_counts(counts)

            print(f"  ✅ Cluster-based decentralized training for {len(rep['cluster_hourly_specific'])} clusters completed.")

        print("✅ Decentralized training completed.")

    # =========================================================================
    # recommend() — Identical to centralized version (reads from pre-computed dicts)
    # =========================================================================

    def recommend(self, hour: int, user_cluster: int = None) -> dict:
        """
        Returns recommendations for a given hour (and optional cluster).
        Identical interface to TimeBasedBaselineRecommender.

        Args:
            hour: Hour of day (0-23)
            user_cluster: Optional cluster ID for cluster-specific recommendations

        Returns:
            Dictionary with 'specific' and 'level_1' recommendations
        """
        if user_cluster is not None and self.use_user_clusters:
            cluster_hour_dict = self.popular_specific_by_hour_and_cluster.get(user_cluster, {})
            rec_specific = cluster_hour_dict.get(
                hour,
                self.global_popular_specific_by_cluster.get(user_cluster, self.global_popular_specific)
            )

            cluster_level1_dict = self.popular_level1_by_hour_and_cluster.get(user_cluster, {})
            rec_level1 = cluster_level1_dict.get(
                hour,
                self.global_popular_level1_by_cluster.get(user_cluster, self.global_popular_level1)
            )
        else:
            rec_specific = self.popular_specific_by_hour.get(hour, self.global_popular_specific)
            rec_level1 = self.popular_level1_by_hour.get(hour, self.global_popular_level1)

        return {
            'specific': rec_specific,
            'level_1': rec_level1
        }
