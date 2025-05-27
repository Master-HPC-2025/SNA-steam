import requests
import time
import json
from collections import deque, defaultdict
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random


class FastDenseSteamNetworkCollector:
    def __init__(self, api_key, request_delay=0.1, max_workers=8):
        """
        Fast + Dense Steam Network Collector
        Optimized for speed while maintaining density focus
        """
        self.api_key = api_key
        self.base_url = "http://api.steampowered.com"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SteamNetworkCollector/1.0'})

        # Core network data
        self.user_map = {}
        self.friendships = set()
        self.user_friends = {}  # Cache friend lists
        self.processed_users = set()
        self.failed_users = set()

        # Dense network optimization
        self.connection_counts = defaultdict(int)  # Track internal connections per user
        self.priority_users = []  # High-connectivity users
        self.community_clusters = defaultdict(set)  # Simple clustering by friend overlap

        # Performance settings
        self.request_delay = request_delay
        self.max_workers = max_workers

        # Thread safety
        self.user_map_lock = threading.Lock()
        self.friendships_lock = threading.Lock()
        self.processed_lock = threading.Lock()

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_user_info_minimal(self, steam_id):
        """Get minimal user info needed for density decisions"""
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        params = {'key': self.api_key, 'steamids': steam_id}

        try:
            response = self.session.get(url, params=params, timeout=3)
            if response.status_code != 200:
                return None

            data = response.json()
            if 'response' in data and 'players' in data['response'] and data['response']['players']:
                player = data['response']['players'][0]
                return {
                    'public': player.get('communityvisibilitystate', 1) == 3,
                    'country': player.get('loccountrycode', ''),
                    'active': player.get('lastlogoff', 0) > time.time() - 30 * 24 * 3600
                }
            return None
        except:
            return None

    def get_friend_list_fast(self, steam_id):
        """Fast friend list with caching"""
        if steam_id in self.user_friends:
            return self.user_friends[steam_id]

        if steam_id in self.failed_users:
            return []

        url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
        params = {'key': self.api_key, 'steamid': steam_id, 'relationship': 'friend'}

        try:
            response = self.session.get(url, params=params, timeout=3)
            if response.status_code == 403:
                self.failed_users.add(steam_id)
                self.user_friends[steam_id] = []
                return []
            elif response.status_code != 200:
                return []

            data = response.json()
            friends = []
            if 'friendslist' in data and 'friends' in data['friendslist']:
                friends = [friend['steamid'] for friend in data['friendslist']['friends']]

            self.user_friends[steam_id] = friends
            return friends
        except:
            self.failed_users.add(steam_id)
            self.user_friends[steam_id] = []
            return []

    def calculate_density_score_fast(self, user_id, friends, user_info=None):
        """Fast density score calculation"""
        if not friends:
            return 0

        score = 0

        # Base score from friend count (prefer moderate counts for density)
        friend_count = len(friends)
        if 10 <= friend_count <= 50:
            score += 50
        elif 50 < friend_count <= 100:
            score += 30
        elif friend_count > 100:
            score += 10  # Penalize super-high degree nodes

        # Bonus for internal connections (friends already in network)
        internal_connections = sum(1 for f in friends if f in self.user_map)
        score += internal_connections * 15  # Major bonus for density

        # Country clustering bonus
        if user_info and user_info.get('country'):
            country = user_info['country']
            if len(self.community_clusters[country]) > 5:  # Existing community
                score += 25

        # Activity bonus
        if user_info and user_info.get('active'):
            score += 10

        return score

    def find_high_density_seeds_fast(self, initial_seeds, sample_size=30):
        """Quickly find high-density seed candidates"""
        candidates = []

        self.logger.info("🔍 Finding density-optimized seeds...")

        def process_seed(seed_id):
            user_info = self.get_user_info_minimal(seed_id)
            if not user_info or not user_info['public']:
                return None

            friends = self.get_friend_list_fast(seed_id)
            if not friends:
                return None

            # Sample friends for density potential
            sample_friends = random.sample(friends, min(sample_size, len(friends)))
            total_score = 0
            valid_samples = 0

            for friend_id in sample_friends:
                friend_info = self.get_user_info_minimal(friend_id)
                if friend_info and friend_info['public']:
                    friend_friends = self.get_friend_list_fast(friend_id)
                    score = self.calculate_density_score_fast(friend_id, friend_friends, friend_info)
                    total_score += score
                    valid_samples += 1

                time.sleep(self.request_delay)

            avg_score = total_score / max(valid_samples, 1)
            return (seed_id, avg_score, len(friends), user_info)

        # Process seeds with threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_seed, seed_id) for seed_id in initial_seeds]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)

        # Sort by density potential
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top density candidates
        selected = [c[0] for c in candidates[:15]]  # Top 15 for density
        self.logger.info(f"🎯 Selected {len(selected)} high-density seeds")

        return selected

    def process_user_batch_dense(self, user_batch):
        """Process batch with density-aware friend selection"""

        def process_single_user_dense(user_data):
            user_id, depth, context_country = user_data

            if user_id in self.failed_users:
                return user_id, [], 0

            friends = self.get_friend_list_fast(user_id)
            if not friends:
                return user_id, [], 0

            # Get minimal user info for density decisions
            user_info = self.get_user_info_minimal(user_id)

            # Score and select best friends for density
            friend_scores = []
            for friend_id in friends:
                if friend_id in self.user_map:
                    # Existing friend - always include for density
                    friend_scores.append((friend_id, 1000, True))
                elif len(self.user_map) < 2500:  # Space available
                    # Potential new friend - score for density
                    friend_info = self.get_user_info_minimal(friend_id)
                    if friend_info and friend_info['public']:
                        friend_friends = self.get_friend_list_fast(friend_id)
                        score = self.calculate_density_score_fast(friend_id, friend_friends, friend_info)

                        # Bonus for same country (clustering)
                        if context_country and friend_info.get('country') == context_country:
                            score += 20

                        friend_scores.append((friend_id, score, False))

                    time.sleep(self.request_delay)

            # Sort by score and select top friends
            friend_scores.sort(key=lambda x: x[1], reverse=True)

            # Limit friends per user based on density strategy
            max_friends = 15 if depth < 2 else 8  # More selective at deeper levels
            selected_friends = friend_scores[:max_friends]

            return user_id, selected_friends, len([f for f in friends if f in self.user_map])

        # Process batch
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single_user_dense, ud): ud[0] for ud in user_batch}

            for future in as_completed(futures):
                user_id = futures[future]
                try:
                    uid, selected_friends, internal_count = future.result()
                    results[uid] = (selected_friends, internal_count)
                except Exception as e:
                    self.logger.debug(f"Error processing {user_id}: {e}")
                    results[user_id] = ([], 0)

        return results

    def add_user_to_network(self, steam_id):
        """Thread-safe user addition"""
        with self.user_map_lock:
            if steam_id not in self.user_map:
                self.user_map[steam_id] = len(self.user_map)
                return True
            return False

    def add_connections_batch_dense(self, connections):
        """Add connections and track density metrics"""
        with self.friendships_lock:
            for user1_id, user2_id in connections:
                if user1_id in self.user_map and user2_id in self.user_map:
                    idx1, idx2 = self.user_map[user1_id], self.user_map[user2_id]
                    connection = (min(idx1, idx2), max(idx1, idx2))
                    if connection not in self.friendships:
                        self.friendships.add(connection)
                        # Track internal connections for density
                        self.connection_counts[user1_id] += 1
                        self.connection_counts[user2_id] += 1

    def collect_dense_network_fast(self, seed_user_ids, max_users=2500, target_avg_degree=6):
        """
        Fast dense network collection with density-first strategy
        """
        self.logger.info(f"🚀 Starting FAST + DENSE network collection")
        self.logger.info(f"🎯 Target: {max_users} users, avg degree: {target_avg_degree}")

        # Phase 1: Enhanced seed selection for density
        density_seeds = self.find_high_density_seeds_fast(seed_user_ids, sample_size=25)

        # Initialize with density-optimized seeds
        queue = deque()
        seed_countries = {}

        for seed_id in density_seeds:
            if self.add_user_to_network(seed_id):
                user_info = self.get_user_info_minimal(seed_id)
                country = user_info.get('country', '') if user_info else ''
                seed_countries[seed_id] = country
                if country:
                    self.community_clusters[country].add(seed_id)
                queue.append((seed_id, 0, country))  # (user_id, depth, context_country)

        self.logger.info(f"🌱 Starting with {len(self.user_map)} density-optimized seeds")

        # Phase 2: Density-aware BFS with batching
        batch_size = 30  # Smaller batches for better density control
        processed_count = 0
        target_connections = max_users * target_avg_degree // 2

        while queue and len(self.user_map) < max_users:
            # Prepare batch
            current_batch = []
            for _ in range(min(batch_size, len(queue))):
                if not queue:
                    break
                user_data = queue.popleft()
                user_id = user_data[0]

                with self.processed_lock:
                    if user_id not in self.processed_users:
                        self.processed_users.add(user_id)
                        current_batch.append(user_data)

            if not current_batch:
                break

            # Process batch with density awareness
            batch_results = self.process_user_batch_dense(current_batch)

            # Add users and connections with density priority
            new_connections = []
            new_users = []

            for user_id, (selected_friends, internal_count) in batch_results.items():
                depth = next((d for uid, d, c in current_batch if uid == user_id), 0)
                context_country = next((c for uid, d, c in current_batch if uid == user_id), '')

                for friend_id, score, is_existing in selected_friends:
                    if is_existing:
                        # Existing user - add connection
                        new_connections.append((user_id, friend_id))
                    elif len(self.user_map) < max_users and score > 25:  # Density threshold
                        # New user with good density potential
                        if self.add_user_to_network(friend_id):
                            new_connections.append((user_id, friend_id))
                            new_users.append(friend_id)
                            if depth < 2:  # Continue expansion
                                queue.append((friend_id, depth + 1, context_country))

                            # Update community clustering
                            if context_country:
                                self.community_clusters[context_country].add(friend_id)

            # Add connections in batch
            if new_connections:
                self.add_connections_batch_dense(new_connections)

            processed_count += len(current_batch)

            # Progress and density check
            current_connections = len(self.friendships)
            current_avg_degree = (current_connections * 2) / len(self.user_map) if len(self.user_map) > 0 else 0

            self.logger.info(f"📊 Users: {len(self.user_map)}, Connections: {current_connections}, "
                             f"Avg degree: {current_avg_degree:.2f}, New: +{len(new_users)}")

            # Early success if we hit target density
            if current_connections >= target_connections:
                self.logger.info(f"🎯 Target density achieved!")
                break

            # Stop if no progress
            if len(new_users) == 0 and len(new_connections) < 5:
                self.logger.info("🛑 Density plateau reached")
                break

        # Final statistics
        final_connections = len(self.friendships)
        final_avg_degree = (final_connections * 2) / len(self.user_map) if len(self.user_map) > 0 else 0
        density = final_connections / (len(self.user_map) * (len(self.user_map) - 1) // 2) if len(
            self.user_map) > 1 else 0

        self.logger.info(f"🎉 FAST + DENSE COLLECTION COMPLETE!")
        self.logger.info(f"📊 Final: {len(self.user_map)} users, {final_connections} connections")
        self.logger.info(f"📈 Average degree: {final_avg_degree:.2f}")
        self.logger.info(f"🔗 Network density: {density:.6f}")

        # Community breakdown
        top_communities = sorted(self.community_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        self.logger.info(f"🌍 Top communities: {[(k, len(v)) for k, v in top_communities]}")

    def save_network_only(self, output_prefix="fast_dense_network"):
        """Save the dense network with minimal overhead"""
        n_users = len(self.user_map)

        if not self.friendships:
            adj_matrix = sp.csr_matrix((n_users, n_users))
        else:
            rows, cols = [], []
            for user1_idx, user2_idx in self.friendships:
                rows.extend([user1_idx, user2_idx])
                cols.extend([user2_idx, user1_idx])

            data = np.ones(len(rows))
            adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_users))

        # Save adjacency matrix
        mtx_filename = f"{output_prefix}.mtx"
        mmwrite(mtx_filename, adj_matrix)
        self.logger.info(f"💾 Dense network saved to {mtx_filename}")

        # Save user mapping
        mapping_filename = f"{output_prefix}_mapping.json"
        with open(mapping_filename, 'w') as f:
            json.dump(self.user_map, f)
        self.logger.info(f"💾 User mapping saved to {mapping_filename}")

        # Dense network statistics
        density = adj_matrix.nnz / (adj_matrix.shape[0] * adj_matrix.shape[1]) if adj_matrix.shape[0] > 0 else 0
        avg_degree = (len(self.friendships) * 2) / len(self.user_map) if len(self.user_map) > 0 else 0

        # Degree distribution for density analysis
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()

        stats_filename = f"{output_prefix}_stats.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"Fast Dense Steam Network\n")
            f.write(f"======================\n")
            f.write(f"Users: {len(self.user_map)}\n")
            f.write(f"Connections: {len(self.friendships)}\n")
            f.write(f"Average degree: {avg_degree:.2f}\n")
            f.write(f"Network density: {density:.6f}\n")
            f.write(f"Min degree: {degrees.min()}\n")
            f.write(f"Max degree: {degrees.max()}\n")
            f.write(f"Median degree: {np.median(degrees):.2f}\n")
            f.write(f"Failed users: {len(self.failed_users)}\n")
            f.write(f"\nTop Communities:\n")
            top_communities = sorted(self.community_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            for country, users in top_communities:
                f.write(f"  {country}: {len(users)} users\n")

        self.logger.info(f"💾 Dense network stats saved to {stats_filename}")
        return mtx_filename, mapping_filename, stats_filename


def main():
    parser = argparse.ArgumentParser(description='Fast Dense Steam Network Collector')
    parser.add_argument('--api-key', required=True, help='Steam Web API key')
    parser.add_argument('--seed-users', required=True, help='Comma-separated Steam IDs')
    parser.add_argument('--max-users', type=int, default=2500, help='Maximum users')
    parser.add_argument('--target-degree', type=int, default=6, help='Target average degree')
    parser.add_argument('--output', default='fast_dense_network', help='Output prefix')
    parser.add_argument('--delay', type=float, default=0.1, help='API delay (seconds)')
    parser.add_argument('--workers', type=int, default=8, help='Concurrent workers')

    args = parser.parse_args()

    seed_user_ids = [uid.strip() for uid in args.seed_users.split(',') if uid.strip()]

    print("⚡🔗 FAST + DENSE Steam Network Collector")
    print("=" * 50)
    print(f"🎯 Target: {args.max_users} users, avg degree {args.target_degree}")
    print(f"⚙️  Workers: {args.workers}")
    print("=" * 50)

    collector = FastDenseSteamNetworkCollector(
        api_key=args.api_key,
        request_delay=args.delay,
        max_workers=args.workers
    )

    start_time = time.time()
    collector.collect_dense_network_fast(
        seed_user_ids=seed_user_ids,
        max_users=args.max_users,
        target_avg_degree=args.target_degree
    )
    end_time = time.time()

    print(f"⏱️  Completed in {end_time - start_time:.2f} seconds")
    collector.save_network_only(args.output)
    print("🎉 Fast + Dense network collection complete!")


if __name__ == "__main__":
    main()