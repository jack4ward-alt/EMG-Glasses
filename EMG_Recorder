"""
EMG Sub-Vocal Signal Recorder
==============================
Records and stores EMG signals from 4 MyoWare sensors via Arduino
for sub-vocal word recognition training.

HOW TO USE:
1. Connect Arduino to your laptop via USB
2. Upload the Arduino sketch (emg_arduino.ino) to your Arduino first
3. Run this script: python emg_recorder.py
4. Follow the on-screen prompts

REQUIREMENTS:
    pip install pyserial numpy pandas matplotlib
"""

import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import time
import os
import sys
from datetime import datetime
from collections import deque

# ─────────────────────────────────────────────
# SETTINGS — edit these if needed
# ─────────────────────────────────────────────
BAUD_RATE       = 115200      # Must match the Arduino sketch
SAMPLE_RATE_HZ  = 500         # Samples per second (set in Arduino sketch)
RECORD_DURATION = 1.0         # Seconds to record per word rep
REST_DURATION   = 1.5         # Seconds of rest between reps
NUM_SENSORS     = 4           # Number of MyoWare sensors
DATA_FILE       = "training_data.csv"   # Where recordings are saved
PLOT_WINDOW     = 200         # Number of samples shown in live plot

# Your word vocabulary — edit this list
WORDS = [
    "yes",
    "no",
]


# ─────────────────────────────────────────────
# FIND ARDUINO PORT AUTOMATICALLY
# ─────────────────────────────────────────────
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(keyword in p.description.lower() for keyword in
               ["arduino", "uno", "ch340", "cp210", "ftdi", "usb serial"]):
            return p.device
    # If not found by name, list all ports and let user choose
    if ports:
        print("\nCould not auto-detect Arduino. Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description}")
        choice = int(input("Enter number of your Arduino port: "))
        return ports[choice].device
    return None


# ─────────────────────────────────────────────
# CONNECT TO ARDUINO
# ─────────────────────────────────────────────
def connect_arduino():
    port = find_arduino_port()
    if not port:
        print("ERROR: No serial ports found. Is the Arduino plugged in?")
        sys.exit(1)
    print(f"\nConnecting to Arduino on {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for Arduino to reset after connection
        ser.flushInput()
        print("Connected.")
        return ser
    except serial.SerialException as e:
        print(f"ERROR: Could not connect — {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
# READ ONE LINE OF SENSOR DATA FROM ARDUINO
# Returns list of NUM_SENSORS float values
# ─────────────────────────────────────────────
def read_sample(ser):
    try:
        line = ser.readline().decode("utf-8").strip()
        values = [float(x) for x in line.split(",")]
        if len(values) == NUM_SENSORS:
            return values
    except (ValueError, UnicodeDecodeError):
        pass
    return None


# ─────────────────────────────────────────────
# RECORD ONE WORD REPETITION
# Returns a list of samples recorded over RECORD_DURATION seconds
# ─────────────────────────────────────────────
def record_rep(ser):
    samples = []
    start = time.time()
    while time.time() - start < RECORD_DURATION:
        sample = read_sample(ser)
        if sample:
            samples.append(sample)
    return samples


# ─────────────────────────────────────────────
# EXTRACT FEATURES FROM A LIST OF SAMPLES
# Converts raw signal into a fixed-size feature vector for ML
# ─────────────────────────────────────────────
def extract_features(samples):
    if not samples:
        return None
    arr = np.array(samples)  # shape: (n_samples, n_sensors)
    features = []
    for sensor_idx in range(NUM_SENSORS):
        channel = arr[:, sensor_idx]
        features.extend([
            np.mean(channel),           # average signal level
            np.std(channel),            # signal variability
            np.max(channel),            # peak value
            np.min(channel),            # minimum value
            np.max(channel) - np.min(channel),  # range
            np.mean(np.abs(np.diff(channel))),   # mean absolute difference (rate of change)
            np.sum(channel ** 2) / len(channel), # RMS power
            np.percentile(channel, 25), # lower quartile
            np.percentile(channel, 75), # upper quartile
        ])
    return features


# ─────────────────────────────────────────────
# SAVE FEATURES TO CSV
# ─────────────────────────────────────────────
def save_to_csv(features, word_label, session_id):
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header on first use
        if not file_exists:
            num_features = len(features)
            header = [f"feature_{i}" for i in range(num_features)]
            header += ["word", "session", "timestamp"]
            writer.writerow(header)
        row = features + [word_label, session_id, datetime.now().isoformat()]
        writer.writerow(row)


# ─────────────────────────────────────────────
# SHOW DATASET SUMMARY
# ─────────────────────────────────────────────
def show_summary():
    if not os.path.isfile(DATA_FILE):
        print("\nNo training data recorded yet.")
        return
    df = pd.read_csv(DATA_FILE)
    print(f"\n{'─'*40}")
    print(f"  TRAINING DATA SUMMARY")
    print(f"{'─'*40}")
    print(f"  Total recordings : {len(df)}")
    print(f"  Words recorded   : {df['word'].nunique()}")
    print(f"\n  Recordings per word:")
    counts = df['word'].value_counts()
    for word, count in counts.items():
        bar = "█" * (count // 2)
        status = "✓ GOOD" if count >= 50 else f"⚠ need {50 - count} more"
        print(f"    {word:<12} {count:>3}  {bar}  {status}")
    print(f"{'─'*40}\n")


# ─────────────────────────────────────────────
# LIVE SIGNAL MONITOR (shows signal before recording)
# ─────────────────────────────────────────────
def live_monitor(ser, duration=5):
    print(f"\nShowing live signal for {duration} seconds...")
    print("Move your jaw/throat to see signals respond.\n")

    buffers = [deque([0.0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
               for _ in range(NUM_SENSORS)]

    fig, axes = plt.subplots(NUM_SENSORS, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Live EMG Signal — Check electrode placement", fontsize=12)
    lines = []
    colours = ["#00ff88", "#00aaff", "#ff6644", "#ffcc00"]

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0a0a0a")
        ax.set_ylim(0, 1023)
        ax.set_ylabel(f"Sensor {i+1}", fontsize=8)
        ax.tick_params(labelsize=7)
        line, = ax.plot(list(buffers[i]), color=colours[i], linewidth=1)
        lines.append(line)

    fig.patch.set_facecolor("#111111")
    plt.tight_layout()

    start = time.time()
    while time.time() - start < duration:
        sample = read_sample(ser)
        if sample:
            for i, val in enumerate(sample):
                buffers[i].append(val)
                lines[i].set_ydata(list(buffers[i]))
        plt.pause(0.001)

    plt.close()


# ─────────────────────────────────────────────
# MAIN RECORDING SESSION
# ─────────────────────────────────────────────
def recording_session(ser):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "═"*50)
    print("  EMG WORD RECORDER")
    print("═"*50)
    print(f"\nWords to record: {', '.join(WORDS)}")

    show_summary()

    print("OPTIONS:")
    print("  [1] Record a word")
    print("  [2] View live signal (check electrode placement)")
    print("  [3] Show data summary")
    print("  [4] Run a full session (all words, X reps each)")
    print("  [5] Quit")

    while True:
        print()
        choice = input("Choose an option [1-5]: ").strip()

        if choice == "1":
            print(f"\nAvailable words:")
            for i, w in enumerate(WORDS):
                print(f"  [{i}] {w}")
            idx = int(input("Choose word number: "))
            word = WORDS[idx]
            reps = int(input(f"How many reps of '{word}'? "))

            print(f"\nRecording '{word}' — {reps} repetitions")
            print("Get ready... recording starts in 3 seconds.")
            time.sleep(3)

            for rep in range(1, reps + 1):
                # Countdown before recording so user is ready
                print(f"\n  Rep {rep}/{reps}")
                for count in [3, 2, 1]:
                    print(f"  {count}...", end="", flush=True)
                    time.sleep(0.8)
                print(f"\n  ██ MOUTH '{word.upper()}' NOW ██", flush=True)
                samples = record_rep(ser)
                features = extract_features(samples)
                if features:
                    save_to_csv(features, word, session_id)
                    print(f"  ✓ Saved ({len(samples)} samples)")
                else:
                    print("  ✗ No signal detected — check electrodes")
                if rep < reps:
                    print(f"  Relax... resting {REST_DURATION}s")
                    time.sleep(REST_DURATION)

            print(f"\n✓ Done. {reps} reps of '{word}' saved to {DATA_FILE}")

        elif choice == "2":
            live_monitor(ser, duration=8)

        elif choice == "3":
            show_summary()

        elif choice == "4":
            reps = int(input("How many reps per word? (50 recommended): "))
            print(f"\nFull session: {len(WORDS)} words × {reps} reps = {len(WORDS)*reps} total recordings")
            print("Starting in 5 seconds — get electrodes in place...")
            time.sleep(5)

            for word in WORDS:
                print(f"\n── '{word.upper()}' ──")
                print(f"Prepare to mouth '{word}'")
                time.sleep(2)

                for rep in range(1, reps + 1):
                    print(f"\n  Rep {rep}/{reps}")
                    for count in [3, 2, 1]:
                        print(f"  {count}...", end="", flush=True)
                        time.sleep(0.8)
                    print(f"\n  ██ MOUTH '{word.upper()}' NOW ██", flush=True)
                    samples = record_rep(ser)
                    features = extract_features(samples)
                    if features:
                        save_to_csv(features, word, session_id)
                        print(f"  ✓ Saved")
                    else:
                        print(f"  ✗ weak signal — check electrodes")
                    if rep < reps:
                        print(f"  Relax... {REST_DURATION}s")
                        time.sleep(REST_DURATION)

            print("\n✓ Full session complete!")
            show_summary()

        elif choice == "5":
            print("\nExiting. Your data is saved in:", DATA_FILE)
            break

        else:
            print("Invalid choice. Enter 1-5.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nEMG Sub-Vocal Recorder")
    print("Connecting to Arduino...")
    ser = connect_arduino()
    try:
        recording_session(ser)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        ser.close()
        print("Serial connection closed.")
