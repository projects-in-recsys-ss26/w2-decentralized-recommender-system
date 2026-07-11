import pandas as pd
import folium
import webbrowser
import os

def plot_user_trajectory(df: pd.DataFrame, user_id: int, output_html: str = "user_trajectory.html"):
    print(f"Creating interactive map for User ID: {user_id}...")
    
    user_df = df[df['user_id'] == user_id].copy()
    
    if user_df.empty:
        print(f"⚠️ No data found for User {user_id}.")
        return
        
    # Timestamp to local time -> WICHTIG: Wende es auf user_df an, nicht auf df!
    user_df['utc_time'] = pd.to_datetime(user_df['utc_time']).dt.tz_localize(None)
    user_df['local_time'] = user_df['utc_time'] + pd.to_timedelta(user_df['timezone_offset'], unit='m')
    
    # Jetzt sortieren
    user_df = user_df.sort_values(by='local_time')
    
    center_lat = user_df['latitude'].mean()
    center_lon = user_df['longitude'].mean()
    
    # Karte initialisieren
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=13,
        tiles='CartoDB positron' 
    )
    
    # ==============================================================
    # EBENE 1: Alle Daten (wird initial ausgeblendet)
    # ==============================================================
    fg_all = folium.FeatureGroup(name="All Data", show=False)
    
    # Route für alle Check-ins
    all_coordinates = list(zip(user_df['latitude'], user_df['longitude']))
    folium.PolyLine(
        locations=all_coordinates,
        color='#a9a9a9', # Grau für die Gesamtroute, damit es nicht zu aufdringlich ist
        weight=2,
        opacity=0.6,
        tooltip="Gesamte Route"
    ).add_to(fg_all)
    
    # Marker für alle Check-ins
    for check_idx, row in enumerate(user_df.itertuples(), 1):
        formatted_time = row.local_time.strftime('%Y-%m-%d %H:%M')
        popup_text = f"<b>No:</b> {check_idx}<br><b>Category:</b> {row.venue_category_name}<br><b>Time:</b> {formatted_time}"
        
        folium.CircleMarker(
            location=(row.latitude, row.longitude),
            radius=5,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=f"Stop {check_idx}: {row.venue_category_name}",
            color='gray',
            fill=True,
            fill_color='gray',
            fill_opacity=0.5
        ).add_to(fg_all)
        
    fg_all.add_to(m)

    # ==============================================================
    # EBENE 2: Die einzelnen Tage
    # ==============================================================
    user_df['date'] = pd.to_datetime(user_df['local_time']).dt.date
    days = sorted(user_df['date'].unique())
    
    print(f"Found {len(days)} days of data: {days}")
    
    feature_groups = {}
    for idx, day in enumerate(days):
        day_df = user_df[user_df['date'] == day]
        
        # Nur der erste Tag ist initial sichtbar
        fg = folium.FeatureGroup(name=f"day_{idx}", show=(idx == 0))
        
        day_coordinates = list(zip(day_df['latitude'], day_df['longitude']))
        folium.PolyLine(
            locations=day_coordinates,
            color='#3388ff',
            weight=3,
            opacity=0.8,
            tooltip=f"Route für {day}"
        ).add_to(fg)
        
        for check_idx, row in enumerate(day_df.itertuples(), 1):
            formatted_time = row.local_time.strftime('%Y-%m-%d %H:%M')
            popup_text = f"<b>Category:</b> {row.venue_category_name}<br><b>Time:</b> {formatted_time}"
            
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=6,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{row.venue_category_name} ({formatted_time})",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.8
            ).add_to(fg)
        
        fg.add_to(m)
        feature_groups[idx] = (fg, day, len(day_df))
    
    # Folium LayerControl einfügen (wird durch unser eigenes Menü per CSS versteckt und ferngesteuert)
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # ==============================================================
    # HTML & JS: Das interaktive Menü
    # ==============================================================
    html = f"""
    <style>
        /* Versteckt das Standard-Leaflet-Menü */
        .leaflet-control-layers {{ display: none !important; }}
        
        #custom-menu {{
            position: fixed; top: 10px; right: 10px; background-color: white; 
            border: 2px solid #ccc; border-radius: 5px; padding: 12px 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15); z-index: 999;
            font-family: Arial, sans-serif;
        }}
        
        #day-navigator {{
            display: flex; align-items: center; gap: 12px; margin-top: 15px;
            padding-top: 15px; border-top: 1px solid #eee;
        }}
        
        .nav-btn {{
            background-color: #007bff; color: white; border: none; padding: 6px 12px;
            border-radius: 3px; cursor: pointer; font-size: 16px; font-weight: bold;
        }}
        .nav-btn:disabled {{ background-color: #ccc; cursor: not-allowed; }}
    </style>
    
    <div id="custom-menu">
        <label style="cursor: pointer; font-size: 14px; font-weight: bold; display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="mode-toggle" onchange="toggleMode()" checked style="width: 16px; height: 16px;">
            Tages-Filter aktivieren
        </label>
        
        <div id="day-navigator">
            <button id="prev-btn" class="nav-btn" onclick="changeDay(-1)">←</button>
            <div id="date-display" style="min-width: 120px; text-align: center; font-size: 13px; line-height: 1.4;"></div>
            <button id="next-btn" class="nav-btn" onclick="changeDay(1)">→</button>
        </div>
    </div>
    
    <script>
        window.dayData = {[str(d) for d in days]};
        window.checkinsData = {[count for _, _, count in feature_groups.values()]};
        window.currentDayIndex = 0;
        
        // Hilfsfunktion: Findet und klickt die versteckten Leaflet-Checkboxen
        function setLayerVisibility(layerName, shouldBeVisible) {{
            const checkboxes = document.querySelectorAll('.leaflet-control-layers input[type="checkbox"]');
            checkboxes.forEach(cb => {{
                const labelText = cb.nextSibling ? cb.nextSibling.textContent.trim() : "";
                if (labelText === layerName) {{
                    if (shouldBeVisible && !cb.checked) cb.click();
                    else if (!shouldBeVisible && cb.checked) cb.click();
                }}
            }});
        }}
        
        // Schaltet zwischen der "Alle Daten"-Ansicht und der Tagesansicht um
        function toggleMode() {{
            const isFiltered = document.getElementById('mode-toggle').checked;
            const nav = document.getElementById('day-navigator');
            
            if (isFiltered) {{
                // Zeige Navigator, blende "All Data" aus, zeige nur den aktuellen Tag
                nav.style.display = 'flex';
                setLayerVisibility('All Data', false);
                for(let i=0; i < window.dayData.length; i++) {{
                    setLayerVisibility(`day_${{i}}`, i === window.currentDayIndex);
                }}
            }} else {{
                // Verstecke Navigator, blende alle Tage aus, zeige "All Data"
                nav.style.display = 'none';
                for(let i=0; i < window.dayData.length; i++) {{
                    setLayerVisibility(`day_${{i}}`, false);
                }}
                setLayerVisibility('All Data', true);
            }}
        }}
        
        // Wechselt den Tag im Filter-Modus
        function changeDay(offset) {{
            const newIndex = window.currentDayIndex + offset;
            if (newIndex < 0 || newIndex >= window.dayData.length) return;
            
            setLayerVisibility(`day_${{window.currentDayIndex}}`, false);
            setLayerVisibility(`day_${{newIndex}}`, true);
            
            window.currentDayIndex = newIndex;
            updateDisplay();
        }}
        
        // Aktualisiert das UI (Datum und Buttons)
        function updateDisplay() {{
            const date = window.dayData[window.currentDayIndex];
            const count = window.checkinsData[window.currentDayIndex];
            document.getElementById('date-display').innerHTML = `<strong>${{date}}</strong><br><small>(${{count}} check-ins)</small>`;
            
            document.getElementById('prev-btn').disabled = window.currentDayIndex === 0;
            document.getElementById('next-btn').disabled = window.currentDayIndex === window.dayData.length - 1;
        }}
        
        // Initialisierung mit kurzer Verzögerung für Leaflet
        setTimeout(() => {{
            updateDisplay();
            toggleMode(); // Erzwingt den initialen Status (Filter an)
        }}, 200);
    </script>
    """
    m.get_root().html.add_child(folium.Element(html))

    # Speichern und öffnen
    m.save(output_html)
    print(f"✅ Map saved successfully at: {output_html}")
    
    full_path = os.path.abspath(output_html)
    print("🌍 Opening map in browser...")
    webbrowser.open(f"file://{full_path}")

    print("Done")