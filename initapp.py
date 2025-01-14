from flask import Flask
from config import load_config,config_get,Config
from database import load_or_create_dataframe


def initialize_app():
    try:
        print("Lade Konfiguration...")
        load_config()
        print("Konfigurationsprüfung...")
        reload_interval = config_get('DEFAULT', 'reload_interval', 60000)
        if reload_interval <= 0:
            raise ValueError("Das Reload-Intervall muss größer als 0 sein.")
        print("Konfiguration erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler während der Initialisierung: {e}")
        raise SystemExit("Initialisierung fehlgeschlagen.")

    # Daten laden oder erstellen
    try:
        load_or_create_dataframe(Config.DEFAULT_CSV_STORAGE_PATH, Config.DEFAULT_CSV_FOLDER)

    except Exception as e:
        print(f"Fehler beim Laden oder Erstellen der Daten: {e}")
        raise SystemExit("Initialisierung der Daten fehlgeschlagen.")


    # Flask-App konfigurieren
    app = Flask("Meteor Project")
    load_config()
    app.config.from_object(Config)
    app.config["reload_interval"] = reload_interval
    app.config["debug"] = config_get('DEFAULT', 'debug')
    print("Initialisierung abgeschlossen.")
    return app
