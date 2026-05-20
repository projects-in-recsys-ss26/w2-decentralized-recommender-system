import pandas as pd
import folium
import webbrowser
import os

def plot_user_trajectory(df: pd.DataFrame, user_id: int, output_html: str = "user_trajectory.html"):
    print(f"Erstelle Karte für User ID: {user_id}...")
    
    user_df = df[df['user_id'] == user_id].copy()
    
    if user_df.empty:
        print(f"⚠️ Keine Daten für User {user_id} gefunden.")
        return
        
    user_df = user_df.sort_values(by='utc_time')
    
    center_lat = user_df['latitude'].mean()
    center_lon = user_df['longitude'].mean()
    
    # 1. FIX FÜR DEN 403 FEHLER:
    # Wir wechseln das Kartenmaterial von OSM zu CartoDB Positron.
    # Das ist kostenlos, braucht keinen API Key und blockiert keine lokalen Dateien.
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=13,
        tiles='CartoDB positron' 
    )
    
    coordinates = list(zip(user_df['latitude'], user_df['longitude']))
    
    folium.PolyLine(
        locations=coordinates,
        color='#3388ff', # Ein etwas moderneres Blau
        weight=2,
        opacity=0.6,
        tooltip="Laufweg"
    ).add_to(m)
    
    for idx, row in enumerate(user_df.itertuples()):
        # Wir formatieren die Zeit einmal sauber und nutzen sie für beides
        formatted_time = row.utc_time.strftime('%Y-%m-%d %H:%M')
        
        popup_text = f"""
        <b>Nr:</b> {idx + 1}<br>
        <b>Kategorie:</b> {row.venue_category_name}<br>
        <b>Zeit:</b> {formatted_time}
        """
        
        # NEU: Die Zeit wird direkt mit in den Hover-Text gepackt
        hover_text = f"Stop {idx + 1}: {row.venue_category_name} ({formatted_time})"
        
        folium.CircleMarker(
            location=(row.latitude, row.longitude),
            radius=6,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=hover_text,  # Jetzt inklusive Zeit!
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.8
        ).add_to(m)
        
    # 6. Karte speichern
    m.save(output_html)
    print(f"✅ Karte erfolgreich gespeichert unter: {output_html}")
    
    # 7. NEU: Karte direkt im Browser öffnen
    # os.path.abspath sorgt dafür, dass der Browser den exakten, vollen Pfad auf deiner Festplatte findet
    full_path = os.path.abspath(output_html)
    print("🌍 Öffne Karte im Browser...")
    
    # Der Zusatz "file://" ist wichtig, damit der Browser weiß, dass es eine lokale Datei ist
    webbrowser.open(f"file://{full_path}")