import logging
import io
import os
import base64
from venv import create
import matplotlib

matplotlib.use('Agg')  # Sicherstellen, dass Agg-Backend verwendet wird
import time
import threading
from threading import Lock
from threading import Thread
import configparser
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio
import asyncio
from threading import Lock

lock = Lock()

from database import get_average_last_24h
from config import Config, config_get


########################################################################################################
#   SETUP
########################################################################################################

def setup_matplotlib_font():
    # Schriftgröße einstellen
    font_size = config_get('PlotSettings', 'font_size', 16)

    if font_size < 12 or font_size > 64:
        logging.warning(f"Ungültige Schriftgröße {font_size}. Standardwert wird verwendet.")
        font_size = max(12, min(64, font_size))

    plt.rcParams.update({'font.size': font_size})
    logging.info(f"Matplotlib: Schriftgröße auf {font_size} gesetzt.")


# Definition von interpolate_color
def interpolate_color(start_color, end_color, factor):
    """
    Interpoliert zwischen zwei RGB-Farben basierend auf einem Faktor (0 bis 1).
    """
    return (
        int(start_color[0] + (end_color[0] - start_color[0]) * factor),
        int(start_color[1] + (end_color[1] - start_color[1]) * factor),
        int(start_color[2] + (end_color[2] - start_color[2]) * factor),
    )


########################################################################################################
#   PLOT CHARTS
########################################################################################################

def generate_chart(chart_func, file_path):
    """
    Universelle Funktion, um ein Diagramm zu erstellen und es als Base64 zurückzugeben.

    Args:
        chart_func (function): Die Funktion, die das spezifische Diagramm erstellt.
        file_name (str): Der Name der CSV-Datei, die verwendet wird.

    Returns:
        str or None: Ein Base64-String des Diagramms oder None bei Fehler.
    """
    with lock:  # Threadsicherheit
        try:
            # Chart erstellen und Base64-Daten holen
            img_base64 = chart_func(file_path)
            if not img_base64:
                print(f"Fehler: `chart_func` hat kein gültiges Bild zurückgegeben.")
                return None  # Fehlerhafte Rückgabe
            return img_base64  # Base64-Daten zurückgeben
        except Exception as e:
            print(f"Fehler in generate_chart: {str(e)}")
            return None  # Fehler aufgetreten


########################################################################################################
#   Zeiger Chart
########################################################################################################
def create_zeiger_chart(file_path):
    """
    Erstellt ein Zeigerdiagramm basierend auf Konfigurationswerten.
    Die Werte für obere und untere Grenze, Anzahl der Intervalle und weitere Parameter
    werden direkt aus der Konfigurationsdatei gelesen.
    """
    # Werte aus der Konfigurationsdatei laden
    obere_grenze = config_get('DEFAULT', 'obere_grenze', 300)  # Standardwert: 300
    untere_grenze = config_get('DEFAULT', 'untere_grenze', 0)  # Standardwert: 0
    anzahl_intervalle = config_get('DEFAULT', 'anzahl_intervalle', 5)  # Standardwert: 5 Intervalle
    axis_font_size = config_get('DEFAULT', 'axis_font_size', 28)
    number_font_size = config_get('DEFAULT', 'number_font_size', 28)
    pio.renderers.default = "cdn"

    # Überprüfen der Parameter
    if anzahl_intervalle <= 0:
        raise ValueError("Die Anzahl der Intervalle muss größer als 0 sein.")
    if obere_grenze <= untere_grenze:
        raise ValueError("Die obere Grenze muss größer als die untere Grenze sein.")

    durchschnitt = get_average_last_24h(file_path)
    # print(f"Durchschnitt: {durchschnitt}")

    # Dynamische Schrittgröße berechnen
    intervall_groesse = (obere_grenze - untere_grenze) / anzahl_intervalle

    # Farbinterpolation vorbereiten
    start_color = (211, 211, 211)  # Hellgrau
    mid_color = (255, 165, 0)  # Orange
    end_color = (255, 0, 0)  # Rot

    steps = []
    for i in range(anzahl_intervalle):
        # Berechnen des Faktors für die Position
        faktor = i / (anzahl_intervalle - 1)

        # Interpolation zwischen Grau → Orange → Rot
        if faktor <= 0.5:
            # Erste Hälfte: von Grau nach Orange
            faktor_normalisiert = faktor / 0.5
            color_rgb = interpolate_color(start_color, mid_color, faktor_normalisiert)
        else:
            # Zweite Hälfte: von Orange nach Rot
            faktor_normalisiert = (faktor - 0.5) / 0.5
            color_rgb = interpolate_color(mid_color, end_color, faktor_normalisiert)

        # Umwandeln von Farbe in hex
        color_hex = f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})"

        # Intervall berechnen
        step_start = untere_grenze + i * intervall_groesse
        step_end = step_start + intervall_groesse

        steps.append({'range': [step_start, step_end], 'color': color_hex})

    # print("versuche zeiger diagram zu erstellen")

    # Diagramm erstellen und Layout anpassen
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=durchschnitt,  # Wird in der Durchschnittsberechnung erstellt
        number={
            'font': {'size': number_font_size}  # Schriftgröße für die Nummer
        },
        gauge={
            'axis': {
                'range': [untere_grenze, obere_grenze],  # Anzeigebereich der Achse
                'tickfont': {'size': axis_font_size}  # Schriftgröße für Achsenbeschriftung
            },
            'bar': {'color': "blue"},
            'steps': steps,
        }
    ))

    # HINTERGRUND MIT TRANSPARENZ DEFINIEREN
    fig.update_layout(
        paper_bgcolor='rgba(157,223,255,0.6)',  # Hellgelber Hintergrund mit 60 % Transparenz
        plot_bgcolor='rgba(157,223,255,0.6)'  # Gleiches für den Plotbereich
    )
    anzeigen_datum = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    title_font_size = int(config_get('DEFAULT', 'title_font_size'))  # Konvertiert zu Integer
    title_padding = int(config_get('DEFAULT', 'title_padding'))  # Konvertiert zu Integer

    fig.update_layout(
        title={
            'text': f"Durchschnitt pro Stunde<br>vom {anzeigen_datum}",  # Manuell Zeilenumbruch mit <br>
            'font': {'size': title_font_size},  # Schriftgröße für den Titel
            'x': 0.5,  # Zentriert den Titel horizontal
            'xanchor': 'center',  # Fixiert die Zentrierung
            'y': 0.9  # Platziert den Titel mit etwas Platz vom oberen Rand
        },
        margin={
            't': title_padding + 30,  # Erhöht den oberen Rand deutlich
            'b': 0  # Unterer Rand bleibt minimal
        },
        width=1000,
        height=800
    )
    # print("test 4")
    # Verwenden Sie 'scale', um die DPI zu beeinflussen
    scale_factor = 2  # Erhöht die Abtastauflösung
    # Diagramm in HTML umwandeln
    img_base64_zeiger = base64.b64encode(pio.to_image(fig, format='png', scale=scale_factor)).decode('utf-8')
    return img_base64_zeiger


########################################################################################################
#   Diagramm Tagesverlauf gestern
########################################################################################################
def create_tagesverlauf_chart(file_path):
    title_padding = config_get("Default", "title_padding")  # 20 als Standardwert
    try:
        # Lese die CSV-Datei ein
        try:
            df = pd.read_csv(file_path, sep=";")
        except FileNotFoundError:
            print(f"Die Datei konnte nicht gefunden werden: {file_path}")
            return 0
        except Exception as e:
            print(f"Ein Fehler ist beim Laden der Datei aufgetreten: {e}")
            return 0
        # Konvertiere die 'Timestamp'-Spalte in ein datetime-Format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Filtere die Daten auf den letzten vollständigen Tag (00 bis 23 Uhr)
        max_date = df['Timestamp'].dt.floor('D').max()  # Letzter vollständiger Tagesbeginn (00:00)
        df_last_day = df[df['Timestamp'].dt.floor('D') == max_date]
        if df_last_day.empty:
            print("Keine Daten für den letzten Tag gefunden.")
            return 0

        # Debugging-Ausgaben
        # print("Konvertierte Timestamps (erste 5 Zeilen):")
        # print(df_last_day.head())
        # print("Erster und letzter Timestamp:", df_last_day['Timestamp'].min(), df_last_day['Timestamp'].max())

        # Diagrammerstellung
        try:
            # Erstelle ein Bar-Plot für 'Anzahl' und einen Liniendiagramm für 'Kritisch' auf einer zweiten Y-Achse
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Hintergrundfarbe zu Transparenz ändern
            fig.patch.set_facecolor('lightgrey')  # Hintergrundfarbe
            fig.patch.set_alpha(0.5)  # Transparenzlevel (0.0 = komplett transparent, 1.0 = keine Transparenz)
            # Setze Hintergrundfarbe für den Plot-Bereich mit Transparenz
            ax1.set_facecolor('lightgrey')  # Plot-Bereich (Hintergrund) auf 'lightblue' setzen
            ax1.patch.set_alpha(0.5)  # Transparenz von 50% einstellen (0.5)

            # x-Werte sind die Stunden
            x_labels = df_last_day['Timestamp'].dt.strftime('%H').tolist()
            # print("x_labels:", x_labels)

            # Zeige jede zweite Beschriftung an für bessere Übersichtlichkeit auf der X-Achse
            n = 2  # Nur jede zweite Stunde anzeigen
            x_positions = range(len(x_labels))  # Positionen für die X-Achse
            plt.xticks(x_positions[::n], x_labels[::n],
                       rotation=45)  # Schrift um 45 Grad gedreht für bessere Lesbarkeit

            # Dynamische berechnung des maximalen y-Wertes für unterschiedliche Zeiten im Jahr mit 5% Aufschlag, sodass nicht der Maximalwert ganz oben am Bildrand hängt
            max_y_value = max(df_last_day['Anzahl'].max(), df_last_day['Kritisch'].max()) * 1.05
            # print("max Y Achse berechnen:", max_y_value)
            # Erstelle die linke Y-Achse für die "Anzahl"
            ax1.bar(x_labels, df_last_day['Anzahl'], color='blue', alpha=0.6, label='Anzahl')
            ax1.set_xlabel("Stunde")
            ax1.set_ylabel("Anzahl", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0,
                         max_y_value)  # Setze den gleichen Maximalwert für ax1, sodass die Kritischen Werte vergleichbar bleiben
            # print("linke achse erstellt", ax1.get_ylim())
            # Erstelle die zweite Y-Achse für "Kritisch"
            ax2 = ax1.twinx()
            ax2.bar(x_labels, df_last_day['Kritisch'], color='red', alpha=0.6, label='Kritisch')
            ax2.set_ylabel("davon Kritisch", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, max_y_value)  # Setze den gleichen Maximalwert für ax2
            # print("rechte achse erstellt", ax2.get_ylim())
            # Extrahiere das gestrige Datum aus den Daten
            anzeigen_datum = df_last_day['Timestamp'].dt.date.iloc[0]
            # print("anzeigen Datum:", anzeigen_datum)
            # Titel und Legende hinzufügen
            plt.title(f"Stündliche Auswertung vom: {anzeigen_datum}", pad=title_padding)
            # ax1.legend(loc="upper left")
            # ax2.legend(loc="upper right")
            plt.tight_layout()

            # Diagramm in Base64 umwandeln (mit 300dpi und figsize 10,6 entsteht ein Bild mit 3000x1800px, kann reduziert werden wenn nötig)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, dpi=300, format='png')
            img_buf.seek(0)
            img_base64_tagesverlauf = base64.b64encode(img_buf.read()).decode('utf-8').strip()
            # print("Länge des Base64-konvertierten Strings:", len(img_base64_tagesverlauf))
            plt.close(fig)
            return img_base64_tagesverlauf
        except Exception as e:
            print(f"Fehler beim Erstellen des Balkendiagramms: {e}")
            return 0

    except Exception as e:
        print(f"Allgemeiner Fehler: {e}")
        return 0


########################################################################################################
#   Diagramm letzen 7 Tage
########################################################################################################
def create_week_chart(file_path):
    title_padding = config_get("Default", "title_padding")
    try:
        # Lese die CSV-Datei ein
        try:
            df = pd.read_csv(file_path, sep=";")
        except FileNotFoundError:
            print(f"Die Datei konnte nicht gefunden werden: {file_path}")
            return 0
        except Exception as e:
            print(f"Ein Fehler ist beim Laden der Datei aufgetreten: {e}")
            return 0
        # Konvertiere die 'Timestamp'-Spalte in ein datetime-Format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Filtere die Daten auf die letzten 7 vollständigen Tage (00 bis 23 Uhr)
        max_date = df['Timestamp'].dt.floor('D').max()  # Letzter vollständiger Tagesbeginn (00:00)
        last_week_start = max_date - pd.Timedelta(days=6)  # Beginn der letzten 7 vollständigen Tage

        df_last_7_days = df[(df['Timestamp'].dt.floor('D') >= last_week_start) &
                            (df['Timestamp'].dt.floor('D') <= max_date)]
        if df_last_7_days.empty:
            print("Keine Daten für die letzten 7 Tage gefunden.")
            return 0

        # Gruppiere die Daten pro Tag und summiere die Werte (datetime bleibt erhalten)
        df_last_7_days['Date'] = df_last_7_days['Timestamp'].dt.floor('D')  # Nur auf Tagesniveau abrunden
        daily_summary = df_last_7_days.groupby('Date').agg({
            'Anzahl': 'sum',  # Summiere die Spalte "Anzahl"
            'Kritisch': 'sum'  # Summiere die Spalte "Kritisch"
        }).reset_index()  # Reset des Index, damit 'Date' wieder eigene Spalte ist

        # Debugging-Ausgabe
        # print("Tägliche Summen der letzten 7 Tage:")
        # print(daily_summary)

        # Diagrammerstellung
        try:
            # Erstelle ein Bar-Plot für 'Anzahl' und einen Liniendiagramm für 'Kritisch' auf einer zweiten Y-Achse
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Hintergrundfarbe mit Transparenz ändern
            fig.patch.set_facecolor('lightgrey')  # Hintergrundfarbe
            fig.patch.set_alpha(0.5)  # Transparenzlevel (0.0 = komplett transparent, 1.0 = keine Transparenz)
            # Setze Hintergrundfarbe für den Plot-Bereich mit Transparenz
            ax1.set_facecolor('lightgrey')  # Plot-Bereich (Hintergrund) auf 'lightblue' setzen
            ax1.patch.set_alpha(0.5)  # Transparenz von 50% einstellen (0.5)

            # x-Werte sind die Stunden
            x_labels = daily_summary['Date'].dt.strftime('%d').tolist()

            n = 1  # jeden Wert anzeigen
            x_positions = range(len(x_labels))  # Positionen für die X-Achse
            plt.xticks(x_positions[::n], x_labels[::n], rotation=45)  # X-Beschriftungen 45 Grad verdreht anzeigen

            # Berechnung des maximalen y-Wertes
            max_y_value = max(daily_summary['Anzahl'].max(), daily_summary['Kritisch'].max()) * 1.05

            # Erstelle die linke Y-Achse für die "Anzahl"
            ax1.bar(x_labels, daily_summary['Anzahl'], color='blue', alpha=0.6, label='Anzahl')
            ax1.set_xlabel("Tag")
            ax1.set_ylabel("Anzahl", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0, max_y_value)  # Setze den gleichen Maximalwert für ax1

            # Erstelle die zweite Y-Achse für "Kritisch"
            ax2 = ax1.twinx()
            ax2.bar(x_labels, daily_summary['Kritisch'], color='red', alpha=0.6, label='Kritisch')
            ax2.set_ylabel("davon Kritisch", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, max_y_value)  # Setze den gleichen Maximalwert für ax2

            # Ermitteln des Start- und Enddatums aus daily_summary
            start_datum = daily_summary['Date'].min()  # Erstes Datum im Bereich (frühestes Datum)
            end_datum = daily_summary['Date'].max()  # Letztes Datum im Bereich (spätestes Datum)

            # Titel und Legende hinzufügen
            plt.title(f"7 Tage Übersicht vom {start_datum.strftime('%Y-%m-%d')} bis {end_datum.strftime('%Y-%m-%d')}",
                      pad=title_padding)
            # ax1.legend(loc="upper left")
            # ax2.legend(loc="upper right")
            plt.tight_layout()

            # Diagramm in Base64 umwandeln
            img_buf = io.BytesIO()
            plt.savefig(img_buf, dpi=300, format='png')
            img_buf.seek(0)
            img_base64_week = base64.b64encode(img_buf.read()).decode('utf-8').strip()
            plt.close(fig)
            return img_base64_week
        except Exception as e:
            print(f"Fehler beim Erstellen des Balkendiagramms: {e}")
            return 0

    except Exception as e:
        print(f"Allgemeiner Fehler: {e}")
        return 0


########################################################################################################
#   Diagramm letzen 30 Tage
########################################################################################################
def create_month_chart(file_path):
    title_padding = config_get("Default", "title_padding")
    try:
        # Lese die CSV-Datei ein
        try:
            df = pd.read_csv(file_path, sep=";")
        except FileNotFoundError:
            print(f"Die Datei konnte nicht gefunden werden: {file_path}")
            return 0
        except Exception as e:
            print(f"Ein Fehler ist beim Laden der Datei aufgetreten: {e}")
            return 0
        # Konvertiere die 'Timestamp'-Spalte in ein datetime-Format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Filtere die Daten auf die letzten 30 vollständigen Tage (00 bis 23 Uhr)
        max_date = df['Timestamp'].dt.floor('D').max()  # Letzter vollständiger Tagbeginn (00:00)
        last_month_start = max_date - pd.Timedelta(days=29)  # Beginn der letzten 30 vollständigen Tage

        df_last_30_days = df[(df['Timestamp'].dt.floor('D') >= last_month_start) &
                             (df['Timestamp'].dt.floor('D') <= max_date)]
        if df_last_30_days.empty:
            print("Keine Daten für die letzten 30 Tage gefunden.")
            return 0

        # Gruppiere die Daten pro Tag und summiere die Werte (datetime bleibt erhalten)
        df_last_30_days['Date'] = df_last_30_days['Timestamp'].dt.floor('D')  # Nur auf Tagesniveau abrunden
        daily_summary = df_last_30_days.groupby('Date').agg({
            'Anzahl': 'sum',  # Summiere die Spalte "Anzahl"
            'Kritisch': 'sum'  # Summiere die Spalte "Kritisch"
        }).reset_index()  # Reset des Index, damit 'Date' wieder eigene Spalte ist

        # Debugging-Ausgabe
        # print("Tägliche Summen der letzten 30 Tage:")
        # print(daily_summary.head())

        # Diagrammerstellung
        try:
            # Erstelle ein Bar-Plot für 'Anzahl' und 'Kritisch' mit zwei y-Achsen:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Hintergrundfarbe mit Transparenz ändern
            fig.patch.set_facecolor('lightgrey')  # Hintergrundfarbe
            fig.patch.set_alpha(0.5)  # Transparenzlevel (0.0 = komplett transparent, 1.0 = keine Transparenz)
            ax1.set_facecolor('lightgrey')  # Hintergrund für den Plot-Bereich
            ax1.patch.set_alpha(0.5)  # Transparenz für den Plot-Bereich einstellen

            # x-Positionen der Balken
            spacing_factor = 1.8  # Größerer Wert = mehr Abstand
            x_positions = [i * spacing_factor for i in
                           range(len(daily_summary['Date']))]  # x-Positionen mit Abstand skalieren

            # Original-Balkenbreite
            bar_width = 1.2  # Breite der Balken bleibt unverändert!

            # X-Beschriftungen mit den Tagen
            x_labels = daily_summary['Date'].dt.strftime('%d').tolist()
            n = 2  # Zeigt jede zweite Beschriftung an (kann nach Bedarf angepasst werden)
            plt.xticks(x_positions[::n], x_labels[::n], rotation=45)  # Nur jede n-te Position anzeigen

            # Berechnung des maximalen y-Wertes basierend auf beiden Datenreihen
            max_y_value = max(daily_summary['Anzahl'].max(), daily_summary['Kritisch'].max()) * 1.05

            # Erstelle die linke Y-Achse für die "Anzahl"
            ax1.bar(x_positions, daily_summary['Anzahl'], width=bar_width, color='blue', alpha=0.6, label='Anzahl')
            ax1.set_xlabel("Tag")
            ax1.set_ylabel("Anzahl", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0, max_y_value)  # Gleicher Maximalwert für ax1

            # Erstelle die rechte Y-Achse für "Kritisch"
            ax2 = ax1.twinx()
            ax2.bar(x_positions, daily_summary['Kritisch'], width=bar_width, color='red', alpha=0.6, label='Kritisch')
            ax2.set_ylabel("davon Kritisch", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, max_y_value)  # Gleicher Maximalwert für ax2
            # Ermitteln des Start- und Enddatums aus daily_summary
            start_datum = daily_summary['Date'].min()  # Erstes Datum im Bereich (frühestes Datum)
            end_datum = daily_summary['Date'].max()  # Letztes Datum im Bereich (spätestes Datum)

            # Titel und Legende hinzufügen
            plt.title(f"30 Tage Übersicht  vom {start_datum.strftime('%Y-%m-%d')} bis {end_datum.strftime('%Y-%m-%d')}",
                      pad=title_padding)
            # ax1.legend(loc="upper left")
            # ax2.legend(loc="upper right")
            plt.tight_layout()

            # Diagramm in Base64 umwandeln
            img_buf = io.BytesIO()
            plt.savefig(img_buf, dpi=300, format='png')
            img_buf.seek(0)
            img_base64_month = base64.b64encode(img_buf.read()).decode('utf-8').strip()

            plt.close(fig)
            return img_base64_month
        except Exception as e:
            print(f"Fehler beim Erstellen des 30 Tage Balkendiagramms: {e}")
            return 0

    except Exception as e:
        print(f"Allgemeiner Fehler: {e}")
        return 0
