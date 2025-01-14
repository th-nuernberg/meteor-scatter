import pandas as pd
import random

# Start- und Enddatum als Variablen definieren
start_date = "2024-11-30"
end_date = "2025-01-15"

# Datumsbereich erstellen
date_range = pd.date_range(start=start_date, end=end_date)

# Für jeden Tag im Datumsbereich ein CSV erstellen
for single_date in date_range:
    # Tagesanfang und Tagesende für den aktuellen Tag
    current_day_start = single_date.strftime("%Y-%m-%d 00:05:00")
    current_day_end = single_date.strftime("%Y-%m-%d 23:05:00")

    # Timestamps für den aktuellen Tag erstellen (stündlich von 00:05 bis 23:05)
    timestamps = pd.date_range(start=current_day_start, end=current_day_end, freq="H")

    # Beispiel-Daten für die Spalten "Anzahl"
    anzahl = [random.randint(0, 120) for _ in range(len(timestamps))]

    # "Kritisch"-Werte berechnen: maximal 50% des entsprechenden "Anzahl"-Wertes
    kritisch = [random.randint(0, max(anzahl[i] // 2, 1)) for i in range(len(anzahl))]

    # DataFrame erstellen
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Anzahl": anzahl,
        "Kritisch": kritisch
    })

    # DataFrame als CSV speichern (Dateiname basierend auf dem Datum)
    file_name = single_date.strftime("%Y%m%d.csv")
    df.to_csv(file_name, sep=";", index=False)
    print(f"Die CSV-Datei für den {single_date.strftime('%Y-%m-%d')} wurde erfolgreich erstellt.")
