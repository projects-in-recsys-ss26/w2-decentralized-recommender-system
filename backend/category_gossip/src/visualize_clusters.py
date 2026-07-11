import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def plot_radar_chart(centroids_df: pd.DataFrame, output_path: str = "statistics/cluster_radar_chart.png"):
    """
    Erstellt einen Radar Chart für die User Cluster Centroids.
    """
    # Exclude the 'cluster' column to get the feature dimensions
    features = [col for col in centroids_df.columns if col != 'cluster']
    num_vars = len(features)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1] # Close the loop

    # Automatisch das Maximum der Features finden um yticks besser zu setzen
    # Da Features Prozentwerte oder Ratios sind, nehmen wir max aus df
    max_val = centroids_df[features].max().max()
    
    if max_val > 1.0:
        max_limit = math.ceil(max_val)
        step = max_limit / 4
        yticks_vals = [step, step*2, step*3, max_limit]
    else:
        max_limit = 1.0
        yticks_vals = [0.2, 0.4, 0.6, 0.8]

    # Farbschema vorbereiten
    colormap = plt.get_cmap('tab20')
    
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    import os
    output_dir = os.path.join(os.path.dirname(output_path), "cluster_radar_charts")
    os.makedirs(output_dir, exist_ok=True)

    # Plot each cluster separately
    for i, row in centroids_df.iterrows():
        cluster_id = int(row['cluster'])
        values = row[features].values.flatten().tolist()
        values += values[:1] # Close the loop
        
        color = colormap(i % 20)
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], features, size=10)
        ax.set_rlabel_position(0)
        
        if max_val > 1.0:
            plt.yticks(yticks_vals, [f"{v:.1f}" for v in yticks_vals], color="grey", size=8)
            plt.ylim(0, max_limit)
        else:
            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
            plt.ylim(0, 1.0)
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster_id}', color=color)
        ax.fill(angles, values, color=color, alpha=0.3)

        plt.title(f"User Cluster {cluster_id} (Feature Space)", size=16, y=1.1)
        plt.tight_layout()
        
        cluster_output_path = os.path.join(output_dir, f"cluster_{cluster_id}.png")
        plt.savefig(cluster_output_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ {len(centroids_df)} Radar Charts erfolgreich unter '{output_dir}' gespeichert!")
