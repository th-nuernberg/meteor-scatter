import csv
import os
import matplotlib
import time
import threading
from threading import Lock, Thread
import configparser
import pandas as pd
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import asyncio

from config import Config, config_get, calculate_last_month, CURRENT_DF

# Funktion: Prüfe und lade final_dataframe aus CSV-Datei oder erstelle eine neue CSV aus CSV_STORAGE_PATH (Tages CSV Dateien), falls keine vorhanden ist
def load_or_create_dataframe(csv_storage_path, folder_path):
    """
    Ladet den final_dataframe aus einer vorhandenen CSV oder erstellt ihn aus Originalquellen.
    :param csv_storage_path: DEFAULT_CSV_STORAGE_PATH Pfad zur gespeicherten CSV-Datei des DataFrames.
    :param folder_path: DEFAULT_CSV_FOLDER Ordnerpfad mit den originalen CSV-Dateien.
    :return: Ein DataFrame, entweder geladen oder neu erstellt.
    """
    # Falls die gespeicherte CSV-Datei existiert, lade sie
    if os.path.exists(csv_storage_path):
        try:
            df = pd.read_csv(csv_storage_path, sep=";", encoding="utf-8", low_memory=False)
            if df.empty or df is None:
                print("Warnung: CSV-Datei existiert, aber der DataFrame ist leer.")
                return None
            return df
        except Exception as e:
            print(f"Fehler beim Laden der CSV: {e}")
            return None
    # Wenn die Datei gelöscht wurde, wird ein neuer DataFrame erzeugt
    print(f"CSV-Datei nicht gefunden. Erstelle neue aus Quellen im Ordner: {folder_path}")
    df = load_last_30_days_csv_files(folder_path)
    if df is None or df.empty:
        print("Warnung: Die Funktion 'load_last_30_days_csv_files' lieferte keinen gültigen DataFrame.")
        return None

    # Speichern und zurückgeben
    save_dataframe(csv_storage_path, df)
    return df


# Funktion: Speichere den DataFrame als CSV
def save_dataframe(csv_storage_path, dataframe):
    """
    Speichert den DataFrame als CSV-Datei.
    :param csv_storage_path: Speicherort für die CSV-Datei.
    :param dataframe: Der zu speichernde DataFrame.
    """
    try:
        # Speichern ohne Anführungszeichen und Escapes
        dataframe.to_csv(csv_storage_path, index=False, sep=";", quoting=csv.QUOTE_NONE, encoding="utf-8")
        print(f"DataFrame erfolgreich in {csv_storage_path} gespeichert!")
    except Exception as e:
        print(f"Fehler beim Speichern des DataFrames: {e}")

# Funktion: Scanne und lade CSV-Dateien aus den letzten vollen 30 Tagen
def load_last_30_days_csv_files(folder_path):
    """
    Lädt CSV-Dateien aus den letzten vollen 30 Tagen und kombiniert sie in einem DataFrame.
    :param folder_path: Pfad zum Ordner, der die CSV-Dateien enthält
    :return: DataFrame mit Daten der letzten vollen 30 Tage
    """
    dataframes = []
    start_date, end_date = calculate_last_month()

    # Liste aller relevanten Dateien aus den letzten vollen 30 Tagen
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    last_30_days_files = []

    #durchsucht die Namen der CSV Files, in dem bereitgestellten Ordner,nach Daten der letzten vollen 30 Tage
    for file_name in all_files:
        # Extrahiere das Datum aus dem Dateinamen (Format: YYYYMMDD.csv)
        date_str = file_name.split('.')[0]  # Nehme den Teil vor der Erweiterung
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            # Nehme nur Dateien im Zeitraum end_date <= file_date <= start_day
            if start_date <= file_date <= end_date:
                last_30_days_files.append(file_name)
        except ValueError:
            print(f"Warnung: Datei {file_name} hat kein akzeptables Datumsformat und wird übersprungen.")
    #print(f"Ausgewählte Dateien der letzten vollen 30 Tage: {last_30_days_files}")

    # Lade die ausgewählten Dateien
    for file_name in last_30_days_files:
        file_path = os.path.join(folder_path, file_name)
        #print(f"Versuche zugriff auf: {file_path}")
        #print(f"Aktuelles Arbeitsverzeichnis: {os.getcwd()}")
        #print(f"Pfad: {file_path} - Existiert: {os.path.exists(file_path)} - Ist Datei: {os.path.isfile(file_path)}")
        try:
            # CSV-Datei laden
            df = pd.read_csv(file_path,sep=";", encoding="utf-8")
            dataframes.append(df)
            print(f"{file_name} erfolgreich geladen!")
        except Exception as e:
            print(f"Fehler beim Laden der Datei {file_name}: {e}")

    # Kombiniere die geladenen DataFrames, falls vorhanden
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        print("Es konnten keine validen Dateien geladen werden.")
        return None


# Funktion, um die Daten komplett neu zu laden, wenn das Datum des letzten Eintrages ungleich gestern ist
def update_csv_if_needed(folder_path, final_dataframe):
        """
        Überprüft, ob das letzte Datum im DataFrame ungleich 'gestern' ist,
        und lädt in diesem Fall alle CSVs neu.
        :param folder_path: Pfad zum CSV-Ordner mit den Dateien
        :param final_dataframe: Der bestehende DataFrame mit geladenen Daten
        :return: Ein (möglicherweise neu geladener) DataFrame
        """
        # Datum von gestern berechnen
        gestern = (datetime.now() - timedelta(days=1)).date()

        final_dataframe = pd.read_csv(Config.DEFAULT_CSV_STORAGE_PATH, sep=";", encoding="utf-8", low_memory=False)
        # Debugging: Spaltennamen ausgeben
        #print("Debugging Spalten nach Einlesen der CSV:")
        #print(final_dataframe.columns.tolist())

        # Prüfen, ob final_dataframe gültig ist und überprüfbar ist
        if final_dataframe is None or not isinstance(final_dataframe, pd.DataFrame):
            #print(final_dataframe.head())
            print("Der übergebene DataFrame ist ungültig oder leer. Alle CSV-Daten werden neu geladen.")
            return load_last_30_days_csv_files(folder_path)

        # Prüfen, ob es eine 'Timestamp'-Spalte gibt
        if 'Timestamp' not in final_dataframe.columns:
            print("Spalte 'Timestamp' fehlt im aktuellen DataFrame. Alle CSV-Daten werden neu geladen.")
            #print("Spalten im DataFrame:", final_dataframe.columns.tolist())
            return load_last_30_days_csv_files(folder_path)

        # Letztes Timestamp im aktuellen DataFrame
        letzte_datum = pd.to_datetime(final_dataframe['Timestamp']).dt.date.max()

        #print(f"Letztes Timestamp im DataFrame: {letzte_datum}")
        #print(f"Gestern war: {gestern}")

        # Prüfen, ob das letzte Timestamp ungleich gestern ist
        if letzte_datum != gestern:
            print("Das letzte Timestamp ist ungleich gestern. Alle CSV-Daten werden neu geladen.")
            return load_last_30_days_csv_files(folder_path)

        print("Das letzte Timestamp ist gleich gestern. Keine Aktion erforderlich.")
        #print("Spalten im DataFrame:", final_dataframe.columns.tolist())
        return final_dataframe


def scheduled_csv_update():
    """
    Geplante Funktion, die automatisch im Intervall ausgeführt wird, um CSV-Dateien zu checken.
    """
    input_folder = Config.DEFAULT_CSV_FOLDER
    output_file = Config.DEFAULT_CSV_STORAGE_PATH

    try:
        if os.path.exists(output_file):
            existing_data = pd.read_csv(Config.DEFAULT_CSV_STORAGE_PATH, sep=";", encoding="utf-8", low_memory=False)  # Existierende CSV laden
            #print("scheduled_csv_update FKT: Spalten im DataFrame:", existing_data.columns.tolist())
        else:
            # Verwenden die load_or_create_dataframe-Funktion falls keine Datei gefunden wurde (usecase: Programm läuft und Datei geht verloren)
            print("CSV nicht gefunden - lade oder erstelle neuen DataFrame.")
            existing_data = load_or_create_dataframe(output_file, input_folder)

        # CSV neu laden, falls erforderlich
        updated_data = update_csv_if_needed(input_folder, existing_data)

        # Aktualisierte DataFrame als CSV speichern
        updated_data.to_csv(output_file, index=False, sep=";", quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
        print(
            f"scheduled_csv_update FKT: Aktualisiertes DataFrame gespeichert als CSV: {output_file} (Spalten: {updated_data.columns.tolist()})"
        )
        #print("scheduled_csv_update FKT aktualisierte DF speichern: Spalten im DataFrame:", existing_data.columns.tolist())

    except Exception as e:
        print(f"Fehler im CSV-Scheduler-Update: {str(e)}")


########################################################################################################
                            #   Durchschnitt gestern 24h
########################################################################################################
def get_average_last_24h(final_dataframe):
    """
    Berechnet den Durchschnittswert der Spalte 'Anzahl'
    für den gesamten gestrigen Tag basierend auf den Daten in `merged_data.csv`.
    """
    #print("Berechne Durchschnitt letzte 24h")
    try:
        # Prüfen, ob die Datei existiert
        if not os.path.exists(final_dataframe):
            print(f"Fehler: Datei {final_dataframe} existiert nicht.")
            return 0  # Rückgabe eines Standardwertes

        # Datei laden
        df = pd.read_csv(final_dataframe, delimiter=";", dtype=str, skip_blank_lines=True)
        #print(df)

        # Überprüfen, ob die erwarteten Spalten vorhanden sind
        if df.empty or 'Anzahl' not in df.columns or 'Timestamp' not in df.columns:
            print("Fehler: Keine Daten gefunden oder benötigte Spalten fehlen.")
            return 0  # Rückgabe eines Standardwertes

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            raise TypeError("Die Spalte 'Timestamp' konnte nicht korrekt in das datetime64[ns]-Format konvertiert werden.")
        # Berechnung des Zeitraums: Gestern von 00:00 bis 23:59
        today = datetime.now().date()
        start_yesterday = today - timedelta(days=1)  # Gestern 00:00
        end_yesterday = today - timedelta(seconds=1)  # Gestern 23:59:59

        #print(f"24h durchschnitt vom {start_yesterday} bis {end_yesterday} :")

        filtered_df = df[(df['Timestamp'] >= pd.Timestamp(start_yesterday)) &
                         (df['Timestamp'] <= pd.Timestamp(end_yesterday))].copy()

        # Überprüfen, ob es für diesen Zeitraum Daten gibt
        if filtered_df.empty:
            print("Keine Daten für den gestrigen Zeitraum gefunden.")
            return 0

        # Durchschnittswert (Mittelwert) der Spalte 'Anzahl' berechnen
        # 'Anzahl' sollte ein numerisches Format haben
        filtered_df['Anzahl'] = pd.to_numeric(filtered_df['Anzahl'], errors='coerce').fillna(0)
        durchschnitt = round(filtered_df['Anzahl'].mean())  # Durchschnitt runden
        durchschnitt = int(durchschnitt)  # Sicherstellen, dass es eine Ganzzahl ist

        # Ergebnis ausgeben
        #print(f"Der Durchschnittswert der Spalte 'Anzahl' für gestern beträgt: {durchschnitt}")
        return durchschnitt

    except Exception as e:
        print(f"Fehler beim Berechnen des Durchschnitts: {e}")
        return 0  # Rückgabe eines Standardwertes, falls ein Fehler auftritt


    # Ordner durchlaufen
def scan_folder(folder_path):
    matching_files = []
    start_date, end_date = calculate_last_month()
    for filename in os.listdir(folder_path):
        # Prüfen, ob die Datei das Format yyyymmdd.csv hat
        if filename.endswith(".csv") and len(filename) == 12:  # z.B. "20231010.csv"
            try:
                # Datum aus dem Dateinamen extrahieren (erste 8 Zeichen)
                file_date = datetime.strptime(filename[:8], "%Y%m%d").date()

                # Prüfen, ob das Datum im Zeitraum liegt
                if start_date <= file_date <= end_date:
                    matching_files.append(filename)
            except ValueError:
                # Falls der Dateiname kein gültiges Datum enthält, einfach überspringen
                continue
    return matching_files

# Auf fehlende Tage prüfen basierend auf der Liste der gefundenen Dateien für Anzeige auf HTML
def check_missing_days(found_files):
    # Zeitraum abrufen
    start_date, end_date = calculate_last_month()

    # Alle Daten im Zeitraum berechnen
    all_days = [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range((end_date - start_date).days + 1)]

    # Aus der Liste gefundener Dateien die Datumswerte extrahieren
    existing_days = [filename[:8] for filename in found_files if len(filename) == 12 and filename.endswith(".csv")]

    # Fehlende Tage berechnen
    missing_days = [day for day in all_days if day not in existing_days]

    # Ergebnis ausgeben
    if len(missing_days) == 0:
        print("Alle Daten sind vorhanden! 😊")
    else:
        print(f"Fehlende Daten ({len(missing_days)} Tage):")
        for missing_day in missing_days:
            print(f" - {missing_day}")

    # Abgleich der Anzahl
    #print(f"\nErwartete Dateien: {len(all_days)}")
    #print(f"Gefundene Dateien: {len(existing_days)}")
    #print(f"Fehlende Dateien: {len(missing_days)}")

    return missing_days