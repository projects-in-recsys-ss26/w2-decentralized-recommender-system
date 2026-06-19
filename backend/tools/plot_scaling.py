import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np

def plot_scaling_results(csv_path="../statistics/scaling_results.csv", output_path="../statistics/scaling_plot.png"):
    # Pfad relativ zum Ausführungsort anpassen, falls nötig
    if not os.path.exists(csv_path):
        # Versuche es aus dem Root-Verzeichnis
        csv_path = "statistics/scaling_results.csv"
        output_path = "statistics/scaling_plot.png"
        if not os.path.exists(csv_path):
            print(f"❌ Datei nicht gefunden: {csv_path}")
            return
    
    # Globale Einstellungen für Schriftart und -größe (ideal für Paper / farbblindenfreundlich)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(14, 8))
    
    # Da das Global Baseline Modell für eine gegebene User-Anzahl unabhängig vom Cluster-k ist,
    # nehmen wir den Durchschnitt pro User-Anzahl, um eine saubere Referenzlinie zu plotten.
    global_df = df.groupby('num_users')['global_hit_k_spec'].mean().reset_index()
    plt.plot(global_df['num_users'], global_df['global_hit_k_spec'], 
             marker='o', linestyle='--', color='black', linewidth=3, markersize=8, label='Global Baseline (Hit@K)')
    
    # Plot für jedes K (Anzahl der User-Cluster)
    if 'k_clusters' in df.columns:
        k_values = sorted(df['k_clusters'].unique())
        # Farbblindenfreundliche qualitative Farbpalette (Wong, 2011)
        cb_colors = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#F0E442']
        colors = [cb_colors[i % len(cb_colors)] for i in range(len(k_values))]
        
        for idx, k in enumerate(k_values):
            df_k = df[df['k_clusters'] == k].sort_values('num_users')
            plt.plot(df_k['num_users'], df_k['cluster_hit_k_spec'], 
                     marker='s', linestyle='-', color=colors[idx], linewidth=2.5, markersize=8, label=f'Edge Cluster (k={k})')
    else:
        # Fallback, falls eine alte CSV-Datei ohne k_clusters geladen wird
        plt.plot(df['num_users'], df['cluster_hit_k_spec'], marker='s', linestyle='-', color='#0072B2', linewidth=2.5, markersize=8, label='Edge Cluster Model (Hit@k)')

    # Styling
    plt.title('Scaling Behavior: Recommender Hit@K vs. User Count\nfor Different Cluster Sizes', pad=20)
    plt.xlabel('Number of Users in Training Set')
    plt.ylabel('Hit Rate @ K (Specific Category) [%]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    # Set x-ticks to the actual number of users tested
    users_ticks = sorted(df['num_users'].unique())
    plt.xticks(users_ticks, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    plot_scaling_results()