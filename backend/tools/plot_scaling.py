import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_scaling_results(csv_path="../statistics/scaling_results_10percent_usercluster.csv", output_path="../statistics/scaling_plot.png"):
    # Pfad relativ zum Ausführungsort anpassen, falls nötig
    if not os.path.exists(csv_path):
        # Versuche es aus dem Root-Verzeichnis
        csv_path = "statistics/scaling_results.csv"
        output_path = "statistics/scaling_plot.png"
        if not os.path.exists(csv_path):
            print(f"❌ Datei nicht gefunden: {csv_path}")
            return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Accuracy @ 1 (Specific Category)
    plt.plot(df['num_users'], df['global_hit_k_spec'], marker='o', linestyle='--', color='#9ca3af', linewidth=2, label='Global Baseline (Acc@1)')
    plt.plot(df['num_users'], df['cluster_hit_k_spec'], marker='s', linestyle='-', color='#2563eb', linewidth=2, label='Edge Cluster Model (Acc@1)')
    
    # Styling
    plt.title('Scaling Behavior: Recommender Accuracy vs. User Count', fontsize=14, pad=15)
    plt.xlabel('Number of Users in Training Set', fontsize=12)
    plt.ylabel('Accuracy @ 1 (Specific Category) [%]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Set x-ticks to the actual number of users tested
    plt.xticks(df['num_users'], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    plot_scaling_results()