import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Pfad zur CSV-Datei relativ zum tools Ordner
CSV_PATH = "../statistics/top_k_results.csv"
OUTPUT_PATH = "../statistics/top_k_plot.png"

def plot_top_k_results():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Fehler: Die Datei '{CSV_PATH}' existiert noch nicht.")
        print("Bitte führe zuerst main.py aus, um die Daten zu generieren.")
        return

    # Daten laden
    df = pd.read_csv(CSV_PATH)
    
    # Seaborn Style aktivieren für schönere Plots
    sns.set_theme(style="whitegrid")
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    
    # Linien für die verschiedenen Metriken zeichnen
    if 'local_cat_hit_rate' in df.columns:
        plt.plot(df['top_k'], df['local_cat_hit_rate'], marker='o', linewidth=2, label='Kategorie Hit Rate', color='#3b82f6')
    
    if 'local_poi_hit_rate_1_per_cat' in df.columns:
        plt.plot(df['top_k'], df['local_poi_hit_rate_1_per_cat'], marker='s', linewidth=2, label='POI Hit (Bester Ort pro Kat.)', color='#10b981')
        
    if 'local_poi_hit_rate_3_per_cat' in df.columns:
        plt.plot(df['top_k'], df['local_poi_hit_rate_3_per_cat'], marker='^', linewidth=2, linestyle='--', label='POI Hit (Beste 3 Orte pro Kat.)', color='#059669')

    # Achsenbeschriftung
    plt.title('Genauigkeit in Abhängigkeit der Anzahl empfohlener Kategorien (Top K)', fontsize=14, pad=15)
    plt.xlabel('Anzahl empfohlener Kategorien (Top K)', fontsize=12)
    plt.ylabel('Hit Rate in %', fontsize=12)
    
    # X-Achsen Ticks an die tatsächlichen Werte anpassen
    plt.xticks(df['top_k'].unique())
    plt.ylim(0, 100) # Prozentuale Skala von 0 bis 100
    
    # Legende und Layout
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    
    # Speichern und Anzeigen
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"✅ Plot erfolgreich erstellt und gespeichert unter: backend/statistics/top_k_plot.png")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_top_k_results()
