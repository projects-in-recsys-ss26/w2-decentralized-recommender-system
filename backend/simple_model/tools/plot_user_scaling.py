import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_user_scaling_results(csv_path="C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\backend\\statistics\\user_scaling_results.csv", output_path="C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\backend\\statistics\\user_scaling_results.png"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.normpath(os.path.join(base_dir, csv_path))
    output_file = os.path.normpath(os.path.join(base_dir, output_path))
    
    if not os.path.exists(csv_file):
        print(f"❌ Datei nicht gefunden: {csv_file}")
        return
        
    # Globale Einstellungen für Schriftart und -größe (ideal für Paper / farbblindenfreundlich)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    df = pd.read_csv(csv_file)
    
    # Sortiere nach fraction, damit der Plot chronologisch von 0 bis 1 geht
    df = df.sort_values('fraction')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    x_axis = df['fraction'] * 100 # In %
    
    # --- Plot 1: Category Hit Rate (Global vs Cluster) ---
    if 'global_hit_k_spec' in df.columns:
        ax1.plot(x_axis, df['global_hit_k_spec'], marker='o', linestyle='--', color='black', linewidth=3, markersize=8, label='Global Baseline (Category Hit Rate)')
    
    if 'cluster_hit_k_spec' in df.columns:
        ax1.plot(x_axis, df['cluster_hit_k_spec'], marker='s', linestyle='-', color='#0072B2', linewidth=3, markersize=8, label='Cluster Model (Category Hit Rate)')
    
    ax1.set_title('Category Hit Rate vs. Training Data Fraction', pad=15)
    ax1.set_xlabel('Fraction of Users in Training Set [%]')
    ax1.set_ylabel('Hit Rate @ K (Specific Category) [%]')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best')
    ax1.set_xticks(x_axis)
    
    # --- Plot 2: POI Hit Rate (Local Retrieval, e.g. 2 Places per Cat) ---
    poi_col = 'local_poi_hit_rate_2_per_cat'
    if poi_col in df.columns:
        ax2.plot(x_axis, df[poi_col], marker='^', linestyle='-', color='#D55E00', linewidth=3, markersize=8, label='Cluster Model (POI Hit Rate)')
    else:
        print(f"⚠️ Warnung: Spalte '{poi_col}' fehlt für den zweiten Plot.")
        
    ax2.set_title('POI Hit Rate (2 Places/Category) vs. Training Data Fraction', pad=15)
    ax2.set_xlabel('Fraction of Users in Training Set [%]')
    ax2.set_ylabel('POI Hit Rate [%]')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best')
    ax2.set_xticks(x_axis)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot erfolgreich gespeichert unter: {output_file}")

if __name__ == "__main__":
    plot_user_scaling_results()
