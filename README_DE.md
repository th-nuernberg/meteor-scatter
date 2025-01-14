# MeteorScatter

Dieses Projekt erzeugt einen Flask Server für die Anzeige der mit dem Partnerprojekt Meteor Dedektion gesammelten Daten zur Detektion von Meteoren.


# Meteor Dedektion

Doku im Ordner "Meteor_Detection" zu finden

# FlaskServer

## Features Flask Server 
Es werden immer die letzten vollen 30 Tage ausgewertet, davon werden dann verschiedene Grafiken erzeugt.
zusammengefasst in einer Slideshow
- Tagesübersicht mit stündlicher Auflösung und Anzeige der Anzahl insgesamt und der kritischen Meteore
- 7-Tage-Übersicht mit je der Tagessumme der Anzahl insgesamt und der kritischen Meteore
- 30-Tage-Übersciht mit je der Tagessumme der Anzahl insgesamt und der kritischen Meteore
nebenstehend ein
- Zeiger-Chart mit dem Druchschnitt aller Meteore über den letzen Tag, um einschätzen zu können ob es gerade viel oder wenig Meteore sind.

Bei jedem Start der app.py wird der Stand des Arbeits-Datensatzes "final_dataframe" überprüft, gegebenfalls neu erstellt oder geupdatet. 
Sollten komplette Datensätze fehlen, so wird die auf der Website angezeigt.
Fehlen nur einzelne Stunden, so wird ein Diagramm mit den verfügbaren Stunden erzeugt.

Weiterhin wird das final_dataframe.csv periodisch auf seinen aktuellen Stand überprüft(config.ini: #Scheduler Aktualisierungszeit 
CSV-Datensatz in Minuten - schedule_interval = 2)
Die Prüfung erfolgt immer nur mit dem letzten Eingrag, ob dieser dem Tag "gestern" entspricht. Falls nicht, wird ein Update durchgeführt.
ebenso die Inhalte der Website neu geladen. (config.ini: #Aktualisierungszeit Website reload - reload_interval = 150000 - #60000 1min / 300000 5min)

## Basisdaten Requirements
Als Basis müssen die Daten für jeden Tag in folgendem Format abgespeichert werden:
Dateiname: YYYYMMDD.csv im Ordner csv_files
Dateiinhalt in folgendem Format:
Timestamp;Anzahl;Kritisch
2024-11-30 00:05:00;101;0
2024-11-30 01:05:00;8;3
2024-11-30 02:05:00;99;21
... 
...

## Projektstruktur
-Python Projekt Ordner
	-csv_files 
		-20241210.csv
		-20241211.csv
		-20241212.csv
		...
	-static
		-css
			-styles.css
		-js
			-script.js
	-templates
		-index.html
	-app.py
	-config.py
	-config.ini
	-database.py
	-initapp.py
	-plot.py
	-(final_dataframe.csv) 

## Erklärung Code 
app.py (für Programmstart ausführen)
Hauptanwendung mit allen app.routes 
IP Adresse und Port werden in folgender Zeile eingestellt.
Line 174: app.run(host="0.0.0.0", port=5000, debug=debug_value)

config.py
hier finden sich alle Fallback Werte und die config_read bzw config_get Funktion


config.ini
hier lassen sich diverse Einstellungen vornehmen, bei Änderungen bitte app.py neu starten und Website neu laden.

database.py
verarbeitet alle nötigen Daten und speichert sie in final_dataframe.csv

initapp.py
zur Initialisierung der Flask App vor dem Start

plot.py  
hier werden alle Diagramme erzeugt
bei Bedarf lassen sich die Chart Auflösungen verändern, um kleinere Daten zu erhalten.
muss bei jedem Chart einzeln angepasst werden.
Diagramm in Base64 umwandeln (mit 300dpi und figsize 10,6 entsteht ein Bild mit 3000x1800px, kann reduziert werden wenn nötig)

## Install

## Debug
folgende Stelle in der config.py verändern, um mehr Debug/Logging Ausgaben zu erhalten

logging.XXXXXXXXXX   ERROR/WARNING/INFO möglich

console = logging.StreamHandler()
console.setLevel(logging.ERROR)  # Konsole für nur `INFO`-Level oder höher aktivieren
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)