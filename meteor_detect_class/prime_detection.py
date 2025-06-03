print("Start Prime Detection Imports...")

import sounddevice
import pyaudio
import wave
import time
import csv
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import struct
import twitchrealtimehandler
import detector_and_classification as detection
import os

print("Prime Detection Imports done...")

# TODO ALERTS

# C_FILE_PATH_OUT = "/home/meteor/Documents/meteor-detection/csv-out/"  # TODO CSV OUT PATH
# C_FILE_PATH_OUT_SPEC = "/home/meteor/Documents/meteor-detection/spec-out/"  # TODO SPEC OUT PATH

C_FILE_PATH_OUT = "/Users/maximilianbundscherer/Desktop/meteor/out/"  # TODO CSV OUT PATH
C_FILE_PATH_OUT_SPEC = "/Users/maximilianbundscherer/Desktop/meteor/specOut/"  # TODO SPEC OUT PATH

# C_MS_SPEC_CUT_FACTOR = 8  # Twitch  # TODO Noise Filter
C_MS_SPEC_CUT_FACTOR = 9  # TL  # TODO Noise Filter

C_MS_CLUSTER_MIN_SAMPLES = 5  # TODO Cluster Filter
C_MS_CLUSTER_EPSILON = 30  # TODO Cluster Filter

C_FILE_PATH_SPEC = "/tmp/spectrogram2.jpg"
C_DISPLAY = True
C_SAMPLE_RATE = 5000
C_SEG_LEN = 30
# C_SEG_LEN = 60 + 20 # TL BURST

# Assert Env
assert os.path.exists(C_FILE_PATH_OUT), f"Path not found: {C_FILE_PATH_OUT}"
assert os.path.exists(C_FILE_PATH_OUT_SPEC), f"Path not found: {C_FILE_PATH_OUT_SPEC}"

# Time Measurement
time_meas_dict = {}


def start_time_meas(idk: str):
    time_meas_dict[idk] = datetime.now()


def end_time_meas(idk: str):
    # and print in seconds
    time = datetime.now() - time_meas_dict[idk]
    print(f"Time for {idk}: {time.total_seconds()} seconds")


import scipy.io.wavfile as wav


class WavSegmentReader:
    def __init__(self, file_path, segment_length, sample_rate):
        self.sample_rate, self.audio_data = wav.read(file_path)
        assert self.sample_rate == sample_rate, f"Sample rate mismatch: WAV={self.sample_rate}, expected={sample_rate}"
        if len(self.audio_data.shape) == 1:
            self.audio_data = np.expand_dims(self.audio_data, axis=1)  # make 2D
        self.segment_length = segment_length
        self.current_index = 0
        self.total_samples = self.audio_data.shape[0]

    def grab(self):
        segment_samples = self.segment_length * self.sample_rate
        if self.current_index + segment_samples > self.total_samples:
            raise StopIteration("WAV file fully processed.")
        segment = self.audio_data[self.current_index:self.current_index + segment_samples]
        self.current_index += segment_samples
        return segment


# Augio Grabber
# audio_grabber = twitchrealtimehandler.TwitchAudioGrabber(
#     twitch_url="https://www.twitch.tv/astronomiemuseum",
#     # twitch_url="https://www.twitch.tv/noway4u_sir",
#     blocking=True,  # wait until a segment is available
#     segment_length=C_SEG_LEN,  # segment length in seconds
#     rate=C_SAMPLE_RATE,  # sampling rate of the audio
#     channels=1,  # number of channels
#     dtype=np.int16  # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
# )


audio_grabber = WavSegmentReader(
    file_path="/Users/maximilianbundscherer/Desktop/meteor/data/test2.wav",  # Pfad zur WAV-Datei
    segment_length=C_SEG_LEN,
    sample_rate=C_SAMPLE_RATE
)


# Proc Spec
def plot_spectrogram(iq_segment, fs, display=True, vmin=10, vmax=30):
    # Um Rauschgrund zu entfernen wird die Rauschleistung berechnet:
    NFFT = 2048
    delta_f = fs / NFFT  # Frequenzaufloesung im Spectrogramm und fuer die Rauschberechnung

    Pxx, freqs, bins, im = plt.specgram(iq_segment[:, 0], Fs=fs, NFFT=NFFT,
                                        noverlap=NFFT // 2)  # 29 40,vmin=29, vmax=40

    # PSD
    # plt.figure(figsize=(10, 5))
    # plt.plot(freqs, 10 * np.log10(Pxx))
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Power/Frequency [dB/Hz]')
    # plt.title('Power Spectral Density')
    # plt.xlim(0, 2000)  # Frequenzbereich von 0 bis 2000 Hz
    # # plt.ylim(-100, 0)  # dB-Bereich von -100 bis 0
    # plt.grid()
    # plt.show()
    # plt.close("all")

    # Frequenzbereich in dem nur Rauschen vorkommt und keine Bursts
    lower_freq = 570  # TL
    upper_freq = 620  # TL

    # lower_freq = 250  # TWITCH
    # upper_freq = 800  # TWITCH

    noise_band = (freqs >= lower_freq) & (freqs <= upper_freq)
    # Bandbreite fuer das Frequenzband
    bandwidth = np.sum(noise_band) * delta_f  # Bandbreite in Hz

    # Gesamte Leistung in diesem Frequenzband
    band_power = np.sum(Pxx[noise_band])  # Summe der Leistung im Frequenzband
    power_density_db_hz = 10 * np.log10(band_power / bandwidth)
    factor = 40 / 23

    # db-Werte in Pxx umrechnen
    Pxx_db = 10 * np.log10(Pxx)  # in dB
    Pxx_db[np.isinf(Pxx_db)] = -np.inf

    print("Recommend vmin", power_density_db_hz / factor + C_MS_SPEC_CUT_FACTOR)

    plt.imshow(Pxx_db, aspect='auto', origin='lower', extent=[bins[0], bins[-1], freqs[0], freqs[-1]],
               vmin=power_density_db_hz / factor + C_MS_SPEC_CUT_FACTOR,

               # vmax=40  # Default TWICH
               
               # vmax=60  # Test TL
               vmax=80  # TL Rec
               )
    # plt.xlabel('Time [s]')
    # plt.ylabel('Frequency [Hz]')
    # plt.title('Spectrogram 25 Seconds')

    # plt.ylim(800, 1200)  # Twitch Frequenzbereich
    plt.ylim(550, 900)  # TL Frequenzbereich

    # plt.colorbar(label='Leistung/Frequenz (dB/Hz)')
    fig = plt.axis('off')
    # plt.tight_layout()
    plt.savefig(C_FILE_PATH_SPEC, format='jpg', bbox_inches='tight', pad_inches=0)
    if display:
        pass
        plt.show()
    # plt.show(block=False)
    del Pxx, Pxx_db, band_power
    plt.close()


# Process Loop

fs = C_SAMPLE_RATE
n_critical = 0
n_non_critical = 0

start_time = datetime.now()  # Startzeit
start_time1 = datetime.now()
save_interval = timedelta(minutes=59.8)  # Intervall von 1h
save_interval2 = timedelta(hours=24)  # Intervall von 24 h

previous_date = datetime.now().strftime('%Y-%m-%d')
# Aktuelles Datum im Format YYYYMMDD
date_string = datetime.now().strftime("%Y%m%d")
separator = ";"
# Dateiname basierend auf dem aktuellen Datum
file_name = f"{C_FILE_PATH_OUT}{date_string}.csv"
columns = ["Timestamp", "Anzahl", "Kritisch"]
df = pd.DataFrame(columns=columns)
# Pruefen, ob die Datei bereits existiert
if not os.path.exists(file_name):
    # Datei existiert nicht, also erstellen
    df.to_csv(file_name, sep=separator, index=False)
    print(f"Datei {file_name} wurde erstellt.")
else:
    print(f"Datei {file_name} existiert bereits.")

while True:
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print()
    print("[INFO] Starte neuen Durchlauf...")
    print(f"Startzeit: {start_time} / Current Date {current_date} / Current Time {log_time_str}\n")

    # time.sleep(2)
    # raise Exception("Test Error")

    # Schritt 1: Audiosegment erfassen
    start_time_meas("grab_audio")
    try:
        print("Starte Audioaufnahme...")
        audio_segment = audio_grabber.grab()
        print("Audioaufnahme abgeschlossen.")
    except Exception as e:
        print(f"Fehler bei der Audioaufnahme: {e}")
        time.sleep(5)  # Wartezeit bei Fehlern
        continue
    print(f"Audiosegment Größe: {audio_segment.shape}")
    if audio_segment.shape[0] != C_SEG_LEN * C_SAMPLE_RATE:
        try:
            print("Fehler: Das Audiosegment ist fehlerhaft. Starte Stream neu...")
            try:
                audio_grabber.terminate()
            except Exception as e:
                print(f"Fehler beim Terminieren des alten Streams: {e}")
            time.sleep(5)
            audio_grabber = twitchrealtimehandler.TwitchAudioGrabber(
                twitch_url="https://www.twitch.tv/astronomiemuseum",
                blocking=True,  # wait until a segment is available
                segment_length=C_SEG_LEN,  # segment length in seconds
                rate=C_SAMPLE_RATE,  # sampling rate of the audio
                channels=1,  # number of channels
                dtype=np.int16  # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
            )

            print("Neuer Stream wurde gestartet.")
        except Exception as e:
            print(f"Fehler beim Neustart des Streams: {e}")
            # TODO HANDLING - now via watchog
            time.sleep(5)
            raise Exception("Fehler beim Neustart des Streams !!!")
        continue
    end_time_meas("grab_audio")

    # Schritt 2: Spektrogramm plotten
    start_time_meas("plot_spectrogram")
    print("Erstelle Spektrogramm...")
    plot_spectrogram(audio_segment, fs, C_DISPLAY)
    print("Spektrogramm wurde erstellt.")
    del audio_segment
    end_time_meas("plot_spectrogram")

    # Schritt 3: Burst-Erkennung und Clusterbildung
    start_time_meas("detect_and_cluster_bursts")
    image_path1 = C_FILE_PATH_SPEC
    print("Starte Burst-Erkennung und Clusterbildung...")
    bursts, unique_labels, burst_positions, critical_bursts, non_critical_bursts = detection.detect_and_cluster_bursts(
        image_path1, display=C_DISPLAY, eps=C_MS_CLUSTER_EPSILON, min_samples=C_MS_CLUSTER_MIN_SAMPLES,
        sample_len_s=C_SEG_LEN)
    print("Burst-Erkennung und Clusterbildung abgeschlossen.")
    end_time_meas("detect_and_cluster_bursts")

    # Schritt 4: Ergebnisse aktualisieren
    n_critical += len(critical_bursts)
    n_non_critical += len(non_critical_bursts)
    print(f"Anzahl kritischer Bursts in dieser Stunde: {n_critical}")
    print(f"Anzahl nicht kritischer Bursts in dieser Stunde: {n_non_critical}")
    if len(critical_bursts) > 0 or len(non_critical_bursts) > 0:
        # Copy spec to out
        print("Kopiere Spektrogramm... (hat etwas detektiert)")
        spec_fp_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        spec_fp = f"{C_FILE_PATH_OUT_SPEC}{spec_fp_timestamp}-{len(critical_bursts)}-{len(non_critical_bursts)}.jpg"
        os.system(f"cp {C_FILE_PATH_SPEC} {spec_fp}")

    # Schritt 5: Ueberpruefen, ob 1 Stunden vergangen ist
    if datetime.now() - start_time >= save_interval:
        print("Speichere Ergebnisse...")
        # Zeitstempel für die CSV-Datei ohne Millisekunden formatieren
        formatted_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        new_row = {"Timestamp": formatted_timestamp, "Anzahl": n_critical + n_non_critical, "Kritisch": n_critical}

        # Bestehende Datei einlesen
        df2 = pd.read_csv(file_name, sep=separator)

        # Neue Zeile als DataFrame erstellen
        new_row_df = pd.DataFrame([new_row])

        # Zusammenfügen des bestehenden DataFrames und der neuen Zeile
        df2 = pd.concat([df2, new_row_df], ignore_index=True)  # Hier wird `pd.concat` verwendet

        # DataFrame speichern
        df2.to_csv(file_name, sep=separator, index=False)

        print(f"Neue Zeile hinzugefügt: {start_time}")

        # Variablen zurücksetzen
        n_critical = 0
        n_non_critical = 0
        start_time = datetime.now()

    # Schritt 6: Ueberpruefen, ob 24 Stunden vergangen sind
    if current_date != previous_date:
        previous_date = current_date
        print("Neuer Tag, lege neues csv-file an...")
        # Aktuelles Datum im Format YYYYMMDD
        date_string = datetime.now().strftime("%Y%m%d")
        # Dateiname basierend auf dem aktuellen Datum
        file_name = f"{C_FILE_PATH_OUT}{date_string}.csv"
        columns = ["Timestamp", "Anzahl", "Kritisch"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_name, sep=separator, index=False)
        print(f"Die CSV-Datei '{file_name}' wurde erfolgreich erstellt.")
        # Zurücksetzen der Summen und Aktualisieren der Startzeit
        n_critical = 0
        n_non_critical = 0
        start_time1 = datetime.now()
        print("Summen zurückgesetzt und Startzeit aktualisiert.\n")
