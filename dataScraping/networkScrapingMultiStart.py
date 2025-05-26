import requests
import time
import json
from collections import defaultdict, deque
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite
import argparse
import logging


class SteamNetworkCollector:
    def __init__(self, api_key, request_delay=1.0):
        """
        Initialize the Steam Network Collector

        Args:
            api_key (str): Steam Web API key
            request_delay (float): Delay between API requests in seconds
        """
        self.api_key = api_key
        self.base_url = "http://api.steampowered.com"
        self.session = requests.Session()
        self.user_map = {}  # Maps Steam ID to index
        self.user_info = {}  # Stores user information
        self.friendships = []  # List of friendship pairs
        self.request_delay = request_delay
        self.processed_users = set()  # Track processed users to avoid duplicates

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_user_info(self, steam_id):
        """
        Get user information from Steam ID

        Args:
            steam_id (str): Steam ID of the user

        Returns:
            dict: User information or None if failed
        """
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        params = {
            'key': self.api_key,
            'steamids': steam_id
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'response' in data and 'players' in data['response'] and data['response']['players']:
                return data['response']['players'][0]
            return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error fetching user info for {steam_id}: {e}")
            return None

    def get_friend_list(self, steam_id):
        """
        Get friend list for a Steam user

        Args:
            steam_id (str): Steam ID of the user

        Returns:
            list: List of friend Steam IDs or empty list if failed
        """
        url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
        params = {
            'key': self.api_key,
            'steamid': steam_id,
            'relationship': 'friend'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'friendslist' in data and 'friends' in data['friendslist']:
                return [friend['steamid'] for friend in data['friendslist']['friends']]
            return []

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error fetching friends for {steam_id}: {e}")
            return []

    def is_profile_public(self, steam_id):
        """
        Check if a Steam profile is public

        Args:
            steam_id (str): Steam ID to check

        Returns:
            bool: True if profile is public, False otherwise
        """
        user_info = self.get_user_info(steam_id)
        if user_info:
            # communityvisibilitystate: 1 = private, 3 = public
            return user_info.get('communityvisibilitystate', 1) == 3
        return False

    def add_user_to_network(self, steam_id):
        """
        Add a user to the network if not already present

        Args:
            steam_id (str): Steam ID to add

        Returns:
            bool: True if user was added, False otherwise
        """
        if steam_id in self.user_map:
            return True

        # Check if profile is public
        if not self.is_profile_public(steam_id):
            self.logger.warning(f"⚠️  Profile {steam_id} is private, skipping")
            return False

        # Get user info
        user_info = self.get_user_info(steam_id)
        if not user_info:
            return False

        # Add user to network
        self.user_map[steam_id] = len(self.user_map)
        self.user_info[steam_id] = user_info

        # Log user addition with nice formatting
        persona_name = user_info.get('personaname', 'Unknown')
        country = user_info.get('loccountrycode', 'Unknown')
        profile_url = user_info.get('profileurl', 'N/A')
        self.logger.info(f"✓ User {len(self.user_map):4d}: {persona_name:<25} | {country:<3} | {steam_id}")

        return True

    def get_community_info(self, steam_id):
        """
        Get additional community information for better community detection

        Args:
            steam_id (str): Steam ID

        Returns:
            dict: Community information
        """
        user_info = self.user_info.get(steam_id, {})
        return {
            'country': user_info.get('loccountrycode', 'Unknown'),
            'state': user_info.get('locstatecode', 'Unknown'),
            'city': user_info.get('loccityid', 'Unknown'),
            'account_created': user_info.get('timecreated', 0),
            'last_logoff': user_info.get('lastlogoff', 0),
            'profile_state': user_info.get('profilestate', 0),
            'persona_state': user_info.get('personastate', 0)
        }

    def collect_network_data(self, seed_user_ids, max_users=200, max_depth=2):
        """
        Collect network data starting from multiple seed users

        Args:
            seed_user_ids (list): List of Steam IDs to start crawling from
            max_users (int): Maximum number of users to collect
            max_depth (int): Maximum depth to crawl
        """
        self.logger.info(f"🚀 Starting network collection from {len(seed_user_ids)} seed users")
        self.logger.info(f"⚙️  Configuration: max {max_users} users, depth {max_depth}, delay {self.request_delay}s")
        self.logger.info(f"═══════════════════════════════════════════════════════════════════")

        # Initialize with seed users
        self.logger.info(f"🌱 Initializing with seed users...")

        # Use separate queues for each depth level to ensure proper BFS traversal
        current_depth_queue = []  # Users to process at current depth
        next_depth_queue = []  # Users to process at next depth

        # Add all seed users first
        valid_seeds = []
        for i, seed_id in enumerate(seed_user_ids):
            if self.add_user_to_network(seed_id):
                current_depth_queue.append((seed_id, 0, f"Seed_{i + 1}"))
                valid_seeds.append(seed_id)
                seed_name = self.user_info[seed_id].get('personaname', 'Unknown')
                self.logger.info(f"✅ Seed {i + 1} added: {seed_name}")
            else:
                self.logger.error(f"❌ Failed to add seed user {seed_id} (profile may be private)")

        if not valid_seeds:
            self.logger.error("❌ No valid seed users found!")
            return

        # Process users depth by depth
        depth_stats = defaultdict(int)  # Track users per depth
        community_stats = defaultdict(set)  # Track which seed led to which users
        current_depth = 0

        while (current_depth_queue or next_depth_queue) and len(
                self.user_map) < max_users and current_depth <= max_depth:

            # If current depth queue is empty, move to next depth
            if not current_depth_queue and next_depth_queue:
                current_depth_queue = next_depth_queue
                next_depth_queue = []
                current_depth += 1

                if current_depth > max_depth:
                    break

                self.logger.info(f"📊 DEPTH {current_depth}: Starting exploration...")

            # Process one user from current depth
            if not current_depth_queue:
                break

            current_id, depth, seed_label = current_depth_queue.pop(0)

            if current_id in self.processed_users:
                continue

            self.processed_users.add(current_id)
            community_stats[seed_label].add(current_id)
            depth_stats[depth] += 1

            # Get friends
            self.logger.info(
                f"🔍 Processing {self.user_info[current_id].get('personaname', 'Unknown')} (depth {depth})...")
            time.sleep(self.request_delay)
            friends = self.get_friend_list(current_id)

            if not friends:
                self.logger.warning(f"⚠️  No friends found for {current_id}")
                continue

            # Process friends
            new_connections = 0
            new_users_added = 0

            for friend_id in friends:
                # Add friend to network if not present and within limits
                if friend_id not in self.user_map and len(self.user_map) < max_users:
                    time.sleep(self.request_delay)
                    if self.add_user_to_network(friend_id):
                        new_users_added += 1
                        # Add to next depth queue for further exploration
                        if depth < max_depth:
                            next_depth_queue.append((friend_id, depth + 1, seed_label))

                # Record friendship if both users are in network
                if friend_id in self.user_map:
                    current_idx = self.user_map[current_id]
                    friend_idx = self.user_map[friend_id]

                    # Check if friendship already exists (undirected)
                    friendship_exists = False
                    for existing_friendship in self.friendships:
                        if (existing_friendship[0] == current_idx and existing_friendship[1] == friend_idx) or \
                                (existing_friendship[0] == friend_idx and existing_friendship[1] == current_idx):
                            friendship_exists = True
                            break

                    if not friendship_exists:
                        self.friendships.append((current_idx, friend_idx))
                        new_connections += 1

            # Log progress for this user
            if new_connections > 0 or new_users_added > 0:
                user_name = self.user_info[current_id].get('personaname', 'Unknown')
                self.logger.info(
                    f"🔗 {user_name}: +{new_users_added} users, +{new_connections} connections (Total: {len(self.user_map)} users, {len(self.friendships)} connections)")

            # Progress update every 10 users
            if len(self.user_map) % 10 == 0:
                self.logger.info(
                    f"📈 Progress: {len(self.user_map)}/{max_users} users, {len(self.friendships)} connections")

        # Final statistics
        self.logger.info(f"🎉 COLLECTION COMPLETE!")
        self.logger.info(f"🎯 Final network: {len(self.user_map)} users, {len(self.friendships)} connections")

        # Community statistics
        self.logger.info(f"📊 COMMUNITY BREAKDOWN:")
        for seed_label, users in community_stats.items():
            self.logger.info(f"   {seed_label}: {len(users)} users")

        # Depth statistics
        self.logger.info(f"📊 DEPTH BREAKDOWN:")
        for d in sorted(depth_stats.keys()):
            self.logger.info(f"   Depth {d}: {depth_stats[d]} users")

        if len(self.user_map) > 1:
            avg_connections = (len(self.friendships) * 2) / len(self.user_map)
            density = (len(self.friendships) * 2) / (len(self.user_map) * (len(self.user_map) - 1))
            self.logger.info(f"📈 Network stats: Avg {avg_connections:.1f} connections/user, density {density:.4f}")

    def create_adjacency_matrix(self):
        """
        Create adjacency matrix from collected friendships

        Returns:
            scipy.sparse matrix: Adjacency matrix
        """
        n_users = len(self.user_map)

        if not self.friendships:
            self.logger.warning("No friendships found, creating empty matrix")
            return sp.csr_matrix((n_users, n_users))

        # Create coordinate lists for sparse matrix
        rows = []
        cols = []

        for user1_idx, user2_idx in self.friendships:
            # Add both directions for undirected graph
            rows.extend([user1_idx, user2_idx])
            cols.extend([user2_idx, user1_idx])

        # Create sparse matrix with ones for friendships
        data = np.ones(len(rows))
        adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_users))

        return adj_matrix

    def save_network_data(self, output_prefix="steam_network"):
        """
        Save network data to files with community detection features

        Args:
            output_prefix (str): Prefix for output files
        """
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix()

        # Save as MTX file
        mtx_filename = f"{output_prefix}.mtx"
        mmwrite(mtx_filename, adj_matrix)
        self.logger.info(f"💾 Adjacency matrix saved to {mtx_filename}")

        # Save user information with community features
        user_info_filename = f"{output_prefix}_users.json"
        ordered_users = [None] * len(self.user_map)

        for steam_id, idx in self.user_map.items():
            user_data = self.user_info[steam_id].copy()
            user_data['steam_id'] = steam_id
            user_data['index'] = idx

            # Add community detection relevant features
            community_info = self.get_community_info(steam_id)
            user_data.update(community_info)

            ordered_users[idx] = user_data

        with open(user_info_filename, 'w', encoding='utf-8') as f:
            json.dump(ordered_users, f, indent=2, ensure_ascii=False)
        self.logger.info(f"💾 User information saved to {user_info_filename}")

        # Save network statistics
        stats_filename = f"{output_prefix}_stats.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"Steam Network Statistics\n")
            f.write(f"========================\n")
            f.write(f"Number of users: {len(self.user_map)}\n")
            f.write(f"Number of friendships: {len(self.friendships)}\n")
            f.write(f"Matrix dimensions: {adj_matrix.shape}\n")
            f.write(f"Matrix density: {adj_matrix.nnz / (adj_matrix.shape[0] * adj_matrix.shape[1]):.6f}\n")
            f.write(f"Average degree: {adj_matrix.nnz / adj_matrix.shape[0]:.2f}\n")

            # Country distribution
            country_dist = defaultdict(int)
            for user_info in self.user_info.values():
                country = user_info.get('loccountrycode', 'Unknown')
                country_dist[country] += 1

            f.write(f"\nCountry Distribution:\n")
            for country, count in sorted(country_dist.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {country}: {count} users\n")

        self.logger.info(f"💾 Network statistics saved to {stats_filename}")

        return mtx_filename, user_info_filename, stats_filename


def main():
    parser = argparse.ArgumentParser(description='Collect Steam friendship network data for community detection')
    parser.add_argument('--api-key', required=True, help='Steam Web API key')
    parser.add_argument('--seed-users', required=True,
                        help='Comma-separated list of Steam IDs to start crawling from')
    parser.add_argument('--max-users', type=int, default=200,
                        help='Maximum number of users to collect')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum crawling depth')
    parser.add_argument('--output', default='steam_network',
                        help='Output file prefix')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls (seconds)')

    args = parser.parse_args()

    # Parse seed users
    seed_user_ids = [uid.strip() for uid in args.seed_users.split(',') if uid.strip()]

    if not seed_user_ids:
        print("❌ Error: No valid seed user IDs provided")
        return

    print("🌐 Steam Network Collector - Community Detection Mode")
    print("=" * 60)
    print(f"🎯 Target: {len(seed_user_ids)} seed users → {args.max_users} total users")
    print("=" * 60)

    # Create collector
    collector = SteamNetworkCollector(args.api_key, args.delay)

    # Collect network data
    start_time = time.time()
    collector.collect_network_data(
        seed_user_ids=seed_user_ids,
        max_users=args.max_users,
        max_depth=args.max_depth
    )
    end_time = time.time()

    print("=" * 60)
    print(f"⏱️  Collection completed in {end_time - start_time:.2f} seconds")

    # Save results
    mtx_file, users_file, stats_file = collector.save_network_data(args.output)

    print(f"\n🎉 Network collection complete!")
    print(f"📁 Files created:")
    print(f"   📊 {mtx_file} (adjacency matrix)")
    print(f"   👥 {users_file} (user information)")
    print(f"   📈 {stats_file} (network statistics)")
    print(f"\n💡 Ready for community detection analysis!")


if __name__ == "__main__":
    main()