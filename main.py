import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# NBA API imports
try:
    from nba_api.stats.endpoints import leaguedashplayerstats
    from nba_api.stats.static import players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("nba_api not installed. Install with: pip install nba_api")

class NBAPlayerSimilarity:
    def __init__(self):
        self.df = None
        self.stats_matrix = None
        self.scaler = StandardScaler()
        
    def fetch_nba_data(self, season='2024-25'):
        """Fetch NBA player statistics using nba_api"""
        print("Fetching NBA player data using nba_api...")
        
        if not NBA_API_AVAILABLE:
            print("nba_api not available. Using sample data instead...")
            self.create_sample_data()
            return False
        
        try:
            # Fetch current season player stats
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                season_type_all_star='Playoffs' # use Playoffs and Regular Season
            )
            
            # Convert to DataFrame
            self.df = stats.get_data_frames()[0]
            
            # Filter players with at least 10 games played for meaningful stats
            self.df = self.df[self.df['GP'] >= 10].reset_index(drop=True)
            
            # Rename columns for consistency
            column_mapping = {
                'PLAYER_NAME': 'PLAYER',
                'FG_PCT': 'FG_PCT',
                'FG3_PCT': 'FG3_PCT',
                'FT_PCT': 'FT_PCT'
            }
            self.df.rename(columns=column_mapping, inplace=True)
            
            print(f"Successfully fetched data for {len(self.df)} players")
            print(f"Season: {season}")
            return True
            
        except Exception as e:
            print(f"Error fetching data from nba_api: {e}")
            print("Using sample data instead...")
            self.create_sample_data()
            return False
    
    def create_sample_data(self):
        """Create sample data with realistic NBA statistics"""
        players = {
            'PLAYER': ['Stephen Curry', 'Damian Lillard', 'Klay Thompson', 'Kyrie Irving', 
                      'Trae Young', 'LeBron James', 'Kevin Durant', 'Luka Doncic',
                      'Donovan Mitchell', 'Devin Booker', 'Jayson Tatum', 'Joel Embiid',
                      'Giannis Antetokounmpo', 'Nikola Jokic', 'Anthony Edwards'],
            'PTS': [26.4, 25.2, 17.9, 25.6, 26.2, 25.7, 27.1, 28.4, 26.6, 27.1, 26.9, 33.1, 31.1, 26.4, 25.9],
            'REB': [4.5, 4.2, 3.3, 5.0, 2.8, 7.3, 6.7, 8.8, 4.4, 4.9, 8.1, 10.2, 11.6, 12.4, 5.4],
            'AST': [5.1, 7.3, 2.2, 5.2, 10.2, 7.3, 5.0, 8.0, 5.1, 6.5, 4.6, 4.2, 5.7, 9.0, 5.1],
            'FG_PCT': [.453, .463, .414, .498, .433, .540, .538, .486, .463, .495, .467, .548, .558, .632, .459],
            'FG3_PCT': [.408, .375, .412, .415, .374, .410, .387, .382, .384, .366, .347, .330, .275, .356, .357],
            'FT_PCT': [.910, .916, .926, .903, .862, .750, .857, .786, .867, .885, .831, .856, .645, .821, .836],
            'STL': [0.9, 0.9, 0.5, 1.3, 1.1, 1.3, 0.9, 1.4, 1.5, 0.8, 1.0, 1.2, 1.2, 1.4, 1.3],
            'BLK': [0.4, 0.3, 0.5, 0.5, 0.1, 0.6, 1.1, 0.5, 0.4, 0.3, 0.6, 1.7, 0.8, 0.9, 0.5],
            'TOV': [2.8, 2.3, 1.8, 2.1, 4.0, 3.5, 3.3, 4.1, 2.7, 2.7, 2.4, 3.4, 3.9, 3.0, 3.3]
        }
        self.df = pd.DataFrame(players)
        print("Using sample data with 15 NBA players")
    
    def prepare_data(self):
        """Prepare and normalize the statistics matrix"""
        stat_columns = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'STL', 'BLK', 'TOV']
        available_cols = [col for col in stat_columns if col in self.df.columns]
        
        # Create stats matrix
        stats_df = self.df[available_cols].copy()
        
        # Handle missing values
        stats_df = stats_df.fillna(stats_df.mean())
        
        # Normalize the data using StandardScaler
        self.stats_matrix = self.scaler.fit_transform(stats_df)
        
        print(f"Prepared statistics matrix with shape: {self.stats_matrix.shape}")
        print(f"Features used: {available_cols}")
        
        return available_cols
    
    def find_similar_players(self, player_name, top_n=10):
        """Find most similar players using cosine similarity"""
        # Find player index
        player_mask = self.df['PLAYER'].str.contains(player_name, case=False, na=False)
        
        if not player_mask.any():
            print(f"Player '{player_name}' not found!")
            return None
        
        player_idx = player_mask.idxmax()
        player_vector = self.stats_matrix[player_idx].reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(player_vector, self.stats_matrix)[0]
        
        # Create results dataframe
        results = pd.DataFrame({
            'Player': self.df['PLAYER'],
            'Similarity': similarities
        })
        
        # Sort by similarity (excluding the player himself)
        results = results[results['Player'] != self.df.loc[player_idx, 'PLAYER']]
        results = results.sort_values('Similarity', ascending=False).head(top_n)
        
        return results, player_idx
    
    def visualize_similarity(self, player_name, top_n=10):
        """Create visualizations for player similarity"""
        results, player_idx = self.find_similar_players(player_name, top_n)
        
        if results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Most Similar Players to {self.df.loc[player_idx, "PLAYER"]}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Similarity scores bar chart
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results)))
        ax1.barh(results['Player'], results['Similarity'], color=colors)
        ax1.set_xlabel('Cosine Similarity Score', fontweight='bold')
        ax1.set_title('Player Similarity Ranking')
        ax1.set_xlim([0, 1])
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Stats comparison radar chart
        ax2 = axes[0, 1]
        self.create_radar_chart(ax2, player_idx, results.index[0], player_name)
        
        # 3. Key stats comparison
        ax3 = axes[1, 0]
        self.create_stats_comparison(ax3, player_idx, results.head(5))
        
        # 4. Similarity heatmap
        ax4 = axes[1, 1]
        self.create_similarity_heatmap(ax4, player_idx, results.head(5))
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def create_radar_chart(self, ax, player_idx, similar_idx, player_name):
        """Create a radar chart comparing two players"""
        categories = ['PTS', 'REB', 'AST', 'FG3_PCT', 'STL']
        available_cats = [cat for cat in categories if cat in self.df.columns]
        
        player_stats = self.df.loc[player_idx, available_cats].values
        similar_stats = self.df.loc[similar_idx, available_cats].values
        
        # Normalize for visualization
        max_vals = self.df[available_cats].max().values
        player_normalized = player_stats / max_vals
        similar_normalized = similar_stats / max_vals
        
        angles = np.linspace(0, 2 * np.pi, len(available_cats), endpoint=False).tolist()
        player_normalized = np.concatenate((player_normalized, [player_normalized[0]]))
        similar_normalized = np.concatenate((similar_normalized, [similar_normalized[0]]))
        angles += angles[:1]
        
        ax.plot(angles, player_normalized, 'o-', linewidth=2, label=self.df.loc[player_idx, 'PLAYER'])
        ax.fill(angles, player_normalized, alpha=0.25)
        ax.plot(angles, similar_normalized, 'o-', linewidth=2, label=self.df.loc[similar_idx, 'PLAYER'])
        ax.fill(angles, similar_normalized, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_cats)
        ax.set_ylim(0, 1)
        ax.set_title('Statistical Profile Comparison')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    def create_stats_comparison(self, ax, player_idx, similar_players):
        """Create a grouped bar chart comparing key stats"""
        stats_to_compare = ['PTS', 'AST', 'REB']
        available_stats = [s for s in stats_to_compare if s in self.df.columns]
        
        players_to_compare = [player_idx] + similar_players.index.tolist()
        player_names = self.df.loc[players_to_compare, 'PLAYER'].values
        
        x = np.arange(len(available_stats))
        width = 0.15
        
        for i, idx in enumerate(players_to_compare):
            values = self.df.loc[idx, available_stats].values
        #   ax.bar(x + i * width, values, width, label=self.df.loc[idx, 'PLAYER'])
            bars = ax.bar(x + i * width, values, width, label=self.df.loc[idx, 'PLAYER'])

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Values')
        ax.set_title('Key Statistics Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(available_stats)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    def create_similarity_heatmap(self, ax, player_idx, similar_players):
        """Create a heatmap of statistical similarities"""
        stats = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT']
        available_stats = [s for s in stats if s in self.df.columns]
        
        players_to_compare = [player_idx] + similar_players.index.tolist()
        
        data = self.df.loc[players_to_compare, available_stats].values
        player_names = [name[:15] for name in self.df.loc[players_to_compare, 'PLAYER'].values]
        
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', 
                   xticklabels=available_stats, yticklabels=player_names,
                   ax=ax, cbar_kws={'label': 'Stat Value'})
        ax.set_title('Statistical Heatmap')

# Main execution
if __name__ == "__main__":
    # Ask user for specific player and season
    user_input = input("Enter player and season (e.g., 'Stephen Curry 2024-25'): ")
    target_name, target_szn = user_input.rsplit(' ', 1)
    
    # Initialize the similarity finder
    finder = NBAPlayerSimilarity()
    
    # Fetch NBA data (will use sample data if API fails)
    finder.fetch_nba_data(season=target_szn)
    
    # Prepare the statistics matrix
    finder.prepare_data()
    
    # Find and visualize players similar to Steph Curry
    print("\n" + "="*50)


    print(f"Finding players most similar to {target_name}...")
    print("="*50 + "\n")
    
    results = finder.visualize_similarity(target_name, top_n=20)
    
    if results is not None:
        print("\nTop 10 Most Similar Players to Stephen Curry:")
        print(results.to_string(index=False))
        print("\nSimilarity score ranges from 0 (completely different) to 1 (identical)")