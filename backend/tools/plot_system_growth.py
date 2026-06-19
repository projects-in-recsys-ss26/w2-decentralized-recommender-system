import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_system_growth(csv_path="../statistics/system_growth_results.csv", output_path="../statistics/system_growth_plot.png"):
    if not os.path.exists(csv_path):
        csv_path = "statistics/system_growth_results.csv"
        output_path = "statistics/system_growth_plot.png"
        if not os.path.exists(csv_path):
            print(f"❌ Datei nicht gefunden: {csv_path}")
            return
    
    # Globale Einstellungen für Schriftart (Times New Roman) und -größe
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(14, 8))
    
    # Referenzlinie: Globale System-Baseline
    global_df = df.groupby('num_users')['global_hit_k_spec'].mean().reset_index()
    plt.plot(global_df['num_users'], global_df['global_hit_k_spec'], 
             marker='o', linestyle='--', color='black', linewidth=3, markersize=8, label='Global Baseline (Cold Start Fallback)')
    
    # Plot für jedes K
    if 'k_clusters' in df.columns:
        k_values = sorted(df['k_clusters'].unique())
        cb_colors = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#F0E442']
        colors = [cb_colors[i % len(cb_colors)] for i in range(len(k_values))]
        
        for idx, k in enumerate(k_values):
            df_k = df[df['k_clusters'] == k].sort_values('num_users')
            plt.plot(df_k['num_users'], df_k['cluster_hit_k_spec'], 
                     marker='s', linestyle='-', color=colors[idx], linewidth=2.5, markersize=8, label=f'Edge Clusters (k={k})')

    plt.title('System Growth Behavior: Blended Hit@K over Global Population\nas Training Base Expands', pad=20)
    plt.xlabel('Number of Profiled Users in System')
    plt.ylabel('Blended Hit Rate @ K (Global Evaluated) [%]')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set x-ticks
    users_ticks = sorted(df['num_users'].unique())
    plt.xticks(users_ticks, rotation=45)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ System Growth Plot erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    plot_system_growth()