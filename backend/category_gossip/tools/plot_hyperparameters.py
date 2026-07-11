import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return pd.DataFrame(json.load(f))
    return None

def plot_k_optimization(df, output_dir):
    if df is None or df.empty:
        print("Keine K-Optimization Daten gefunden.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Plot Category Hit Rates
    sns.lineplot(data=df, x='k', y='cluster_acc_1_spec', marker='o', label='Cat Accuracy @ 1')
    sns.lineplot(data=df, x='k', y='cluster_hit_k_spec', marker='s', label='Cat Hit Rate @ K')
    
    # Plot POI Hit Rates
    if 'local_poi_hit_rate_3_per_cat' in df.columns:
        sns.lineplot(data=df, x='k', y='local_poi_hit_rate_3_per_cat', marker='^', label='POI Hit Rate (Top 3)')
        
    plt.title('Hyperparameter Optimization: k (Number of User Clusters)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Hit Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'k_optimization.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    plt.close()

def plot_top_k_optimization(df, output_dir):
    if df is None or df.empty:
        print("Keine Top-K Daten gefunden.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Plot Category Hit Rates
    sns.lineplot(data=df, x='top_k', y='cluster_hit_k_spec', marker='s', label='Cat Hit Rate @ K')
    
    # Plot POI Hit Rate for Top 3 places as baseline
    if 'local_poi_hit_rate_3_per_cat' in df.columns:
        sns.lineplot(data=df, x='top_k', y='local_poi_hit_rate_3_per_cat', marker='^', label='POI Hit Rate (3 Places/Cat)')
        
    plt.title('Hyperparameter Optimization: top_k (Predicted Categories)')
    plt.xlabel('Number of Predicted Categories (top_k)')
    plt.ylabel('Hit Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'top_k_optimization.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    plt.close()

def plot_top_places_tradeoff(df, output_dir):
    """
    Shows how the POI Hit Rate scales with the number of top places per category,
    specifically checking the constraint top_k * top_places <= 10.
    """
    if df is None or df.empty:
        return
        
    plt.figure(figsize=(12, 7))
    
    # Define available top_places from the columns
    top_places_cols = [c for c in df.columns if c.startswith('local_poi_hit_rate_') and c.endswith('_per_cat')]
    if not top_places_cols:
        return
        
    # Extract integer values for places
    places = sorted([int(c.split('_')[4]) for c in top_places_cols])
    
    # We plot the curve for a few interesting top_k values (e.g. 1, 2, 3, 5)
    target_top_ks = [1, 2, 3, 4, 5]
    
    for tk in target_top_ks:
        row = df[df['top_k'] == tk]
        if row.empty:
            continue
            
        row = row.iloc[0]
        y_vals = []
        x_vals = []
        for p in places:
            # ONLY plot points where top_k * top_places <= 10 as per user constraint
            if tk * p <= 10:
                col = f'local_poi_hit_rate_{p}_per_cat'
                if col in row:
                    y_vals.append(row[col])
                    x_vals.append(p)
                    
        if x_vals:
            sns.lineplot(x=x_vals, y=y_vals, marker='o', label=f'top_k={tk} (Total <= {tk*max(x_vals)})')

    plt.title('Trade-off: Places per Category vs POI Hit Rate (Constraint: Total Places <= 10)')
    plt.xlabel('Number of Retrieved Places per Category')
    plt.ylabel('POI Hit Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Predicted Categories')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'top_places_tradeoff.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    stats_dir = os.path.join(base_dir, 'statistics')
    plots_dir = os.path.join(base_dir, 'statistics', 'plots')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    k_df = load_data(os.path.join(stats_dir, 'k_optimization_results.json'))
    plot_k_optimization(k_df, plots_dir)
    
    top_k_df = load_data(os.path.join(stats_dir, 'top_k_results.json'))
    plot_top_k_optimization(top_k_df, plots_dir)
    plot_top_places_tradeoff(top_k_df, plots_dir)
    
    print("\nVisualization complete! Plots are available in:", plots_dir)
