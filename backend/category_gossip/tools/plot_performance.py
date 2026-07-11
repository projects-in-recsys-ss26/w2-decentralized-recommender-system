import matplotlib.pyplot as plt
import os
import json

def plot_performance_snapshot(standard_results_path="statistics/standard_evaluation.json", output_path="statistics/performance_snapshot.png"):
    """
    Erstellt ein Bar Chart für die initiale Performance (Category Hit Rate und POI Hit Rate).
    """
    if not os.path.exists(standard_results_path):
        print(f"⚠️ {standard_results_path} nicht gefunden.")
        return

    with open(standard_results_path, 'r') as f:
        results = json.load(f)

    # Extrahiere die Metriken
    cat_hit = results.get('cluster_hit_k_spec', results.get('global_hit_k_spec', 0))
    
    # Finde die POI Metrik (wir nehmen z.B. die erste verfügbare oder eine spezifische)
    poi_hit = 0
    for key, value in results.items():
        if key.startswith('local_poi_hit_rate_'):
            poi_hit = value
            break # Nimm die erste gefundene (z.B. für N=2)

    labels = ['Category Hit Rate (%)', 'POI Hit Rate (%)']
    values = [cat_hit, poi_hit]
    colors = ['#0072B2', '#D55E00']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, width=0.5)

    plt.title('Model Performance (Leave-One-Out Test)', fontsize=16, pad=15)
    plt.ylabel('Hit Rate [%]', fontsize=14)
    plt.ylim(0, max(100, max(values) * 1.2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Werte über den Balken anzeigen
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Performance Snapshot gespeichert unter: {output_path}")

def plot_learning_curve(fractions, cat_hits, poi_hits, global_cat_hits=None, global_poi_hits=None, decentralized_cat_dict=None, decentralized_poi_dict=None, output_path="statistics/learning_curve.png"):
    """
    Erstellt zwei Line Charts (Category und POI), die zeigen, wie das Modell mit mehr Daten besser wird.
    Beinhaltet einen optionalen Vergleich mit den rein globalen Metriken und mehreren dezentralen Varianten.
    
    Args:
        decentralized_cat_dict: Optional dict {gossip_rounds: [cat_hits_per_fraction]}
        decentralized_poi_dict: Optional dict {gossip_rounds: [poi_hits_per_fraction]}
    """
    x_percentages = [f * 100 for f in fractions]
    
    # Color palette and markers for multiple decentralized curves (high contrast colorblind friendly)
    dec_styles = [
        {'color': '#D55E00', 'marker': 'D'},  # Vermilion
        {'color': '#009E73', 'marker': 'P'},  # Bluish Green
        {'color': '#CC79A7', 'marker': 'X'},  # Reddish Purple
        {'color': '#E69F00', 'marker': 'v'},  # Orange
        {'color': '#56B4E9', 'marker': 'h'},  # Sky Blue
    ]
    
    # 1. Category Hit Rate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_percentages, cat_hits, marker='o', linestyle='-', color='#000000', linewidth=2.5, markersize=8, label='Centralized Cluster-Based')
    if global_cat_hits is not None:
        plt.plot(x_percentages, global_cat_hits, marker='^', linestyle='--', color='#0072B2', linewidth=2, markersize=8, label='Centralized Global-Only')
    if decentralized_cat_dict is not None:
        for idx, (rounds, hits) in enumerate(sorted(decentralized_cat_dict.items())):
            style = dec_styles[idx % len(dec_styles)]
            plt.plot(x_percentages, hits, marker=style['marker'], linestyle='-.', color=style['color'],
                     linewidth=2, markersize=8, label=f'Decentralized ({rounds} Gossip Rounds)')
    
    plt.title('Learning Curve: Category Hit Rate', fontsize=16, pad=15)
    plt.xlabel('Training Data Used [% of Total Train Set]', fontsize=14)
    plt.ylabel('Category Hit Rate [%]', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.xticks(x_percentages)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cat_path = output_path.replace('.png', '_category.png')
    plt.tight_layout()
    plt.savefig(cat_path, dpi=300)
    plt.close()
    
    # 2. POI Hit Rate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_percentages, poi_hits, marker='s', linestyle='-', color='#000000', linewidth=2.5, markersize=8, label='Centralized Cluster-Based')
    if global_poi_hits is not None:
        plt.plot(x_percentages, global_poi_hits, marker='d', linestyle='--', color='#0072B2', linewidth=2, markersize=8, label='Centralized Global-Only')
    if decentralized_poi_dict is not None:
        for idx, (rounds, hits) in enumerate(sorted(decentralized_poi_dict.items())):
            style = dec_styles[idx % len(dec_styles)]
            plt.plot(x_percentages, hits, marker=style['marker'], linestyle='-.', color=style['color'],
                     linewidth=2, markersize=8, label=f'Decentralized ({rounds} Gossip Rounds)')
        
    plt.title('Learning Curve: POI Hit Rate', fontsize=16, pad=15)
    plt.xlabel('Training Data Used [% of Total Train Set]', fontsize=14)
    plt.ylabel('POI Hit Rate [%]', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.xticks(x_percentages)
    
    poi_path = output_path.replace('.png', '_poi.png')
    plt.tight_layout()
    plt.savefig(poi_path, dpi=300)
    plt.close()
    
    # 3. Export Data to CSV
    import pandas as pd
    
    cat_df = pd.DataFrame({
        'Training_Data_Fraction': fractions,
        'Centralized_Cluster_Based': cat_hits,
    })
    if global_cat_hits is not None:
        cat_df['Centralized_Global_Only'] = global_cat_hits
    if decentralized_cat_dict is not None:
        for rounds, hits in sorted(decentralized_cat_dict.items()):
            cat_df[f'Decentralized_{rounds}_Rounds'] = hits
    
    cat_csv_path = output_path.replace('.png', '_category.csv')
    cat_df.to_csv(cat_csv_path, index=False)
    
    poi_df = pd.DataFrame({
        'Training_Data_Fraction': fractions,
        'Centralized_Cluster_Based': poi_hits,
    })
    if global_poi_hits is not None:
        poi_df['Centralized_Global_Only'] = global_poi_hits
    if decentralized_poi_dict is not None:
        for rounds, hits in sorted(decentralized_poi_dict.items()):
            poi_df[f'Decentralized_{rounds}_Rounds'] = hits
            
    poi_csv_path = output_path.replace('.png', '_poi.csv')
    poi_df.to_csv(poi_csv_path, index=False)
    
    print(f"✅ Learning Curves gespeichert unter:\n  - {cat_path}\n  - {poi_path}\n  - {cat_csv_path}\n  - {poi_csv_path}")

