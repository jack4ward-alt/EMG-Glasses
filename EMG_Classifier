"""
EMG Sub-Vocal Word Classifier
==============================
Trains on your recorded CSV data and then listens live
to recognise YES or NO from your EMG sensors.

HOW TO USE:
1. Make sure you have recorded enough data with emg_recorder.py first
   (at least 50 reps per word recommended)
2. Run this script: python classifier.py
3. Choose option 1 to train, then option 2 to go live

REQUIREMENTS:
    pip install pyserial numpy pandas matplotlib scikit-learn
"""

import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import time
import os
import sys
import pickle
from datetime import datetime

# ─────────────────────────────────────────────
# SETTINGS — must match emg_recorder.py
# ─────────────────────────────────────────────
BAUD_RATE        = 115200
NUM_SENSORS      = 4
RECORD_DURATION  = 1.0       # Must match recorder
DATA_FILE        = "training_data.csv"
MODEL_FILE       = "emg_model.pkl"   # Trained model gets saved here
CONFIDENCE_MIN   = 0.70      # Only print word if model is this confident (0-1)


# ─────────────────────────────────────────────
# FIND AND CONNECT ARDUINO
# ─────────────────────────────────────────────
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(k in p.description.lower() for k in
               ["arduino", "uno", "ch340", "cp210", "ftdi", "usb serial"]):
            return p.device
    if ports:
        print("\nCould not auto-detect Arduino. Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description}")
        choice = int(input("Enter number of your Arduino port: "))
        return ports[choice].device
    return None


def connect_arduino():
    port = find_arduino_port()
    if not port:
        print("ERROR: No serial ports found. Is the Arduino plugged in?")
        sys.exit(1)
    print(f"Connecting to Arduino on {port}...")
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)
    ser.flushInput()
    print("Connected.\n")
    return ser


# ─────────────────────────────────────────────
# READ ONE SAMPLE FROM ARDUINO
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
# EXTRACT FEATURES — must match emg_recorder.py
# ─────────────────────────────────────────────
def extract_features(samples):
    if not samples:
        return None
    arr = np.array(samples)
    features = []
    for sensor_idx in range(NUM_SENSORS):
        channel = arr[:, sensor_idx]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.max(channel),
            np.min(channel),
            np.max(channel) - np.min(channel),
            np.mean(np.abs(np.diff(channel))),
            np.sum(channel ** 2) / len(channel),
            np.percentile(channel, 25),
            np.percentile(channel, 75),
        ])
    return features


# ─────────────────────────────────────────────
# TRAIN THE MODEL
# ─────────────────────────────────────────────
def train_model():
    if not os.path.isfile(DATA_FILE):
        print(f"\nERROR: No training data found ({DATA_FILE})")
        print("Record some data with emg_recorder.py first.")
        return None, None

    print(f"\nLoading training data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Show what we have
    print(f"\nTraining data summary:")
    counts = df["word"].value_counts()
    for word, count in counts.items():
        status = "✓" if count >= 50 else f"⚠ only {count} reps (50+ recommended)"
        print(f"  {word:<12} {count} reps  {status}")

    if len(counts) < 2:
        print("\nERROR: Need at least 2 words to train a classifier.")
        return None, None

    # Prepare data
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values
    y = df["word"].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)}...")

    # Build pipeline: scale features then SVM classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, C=10, gamma="scale")),
    ])

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100

    print(f"\n{'═'*40}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═'*40}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"\n  Detailed results:")
    print(classification_report(y_test, y_pred, target_names=sorted(set(y))))

    if accuracy < 70:
        print("  ⚠ Accuracy below 70% — consider recording more reps")
        print("    or checking electrode placement consistency.")
    elif accuracy < 85:
        print("  ✓ Decent accuracy — more reps will improve this further.")
    else:
        print("  ✓✓ Great accuracy — ready for live recognition!")

    # Save model
    words = sorted(set(y))
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "words": words}, f)
    print(f"\n  Model saved to {MODEL_FILE}")

    return model, words


# ─────────────────────────────────────────────
# LOAD SAVED MODEL
# ─────────────────────────────────────────────
def load_model():
    if not os.path.isfile(MODEL_FILE):
        print(f"\nNo saved model found ({MODEL_FILE})")
        print("Train the model first using option 1.")
        return None, None
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Model loaded. Words: {data['words']}")
    return data["model"], data["words"]


# ─────────────────────────────────────────────
# LIVE RECOGNITION
# ─────────────────────────────────────────────
def live_recognition(ser, model, words):
    print(f"\n{'═'*50}")
    print(f"  LIVE RECOGNITION ACTIVE")
    print(f"  Words: {' / '.join(w.upper() for w in words)}")
    print(f"  Confidence threshold: {CONFIDENCE_MIN*100:.0f}%")
    print(f"  Press CTRL+C to stop")
    print(f"{'═'*50}\n")

    # Detection state
    WINDOW_SIZE   = int(RECORD_DURATION * 500)  # samples per window
    STEP_SIZE     = int(WINDOW_SIZE * 0.25)      # slide window by 25%
    buffer        = []
    last_word     = None
    last_word_time = 0
    COOLDOWN      = 1.5   # seconds between detections (prevents repeats)

    print("Listening... mouth a word when ready.\n")

    try:
        while True:
            sample = read_sample(ser)
            if sample is None:
                continue

            buffer.append(sample)

            # Once we have enough samples, run classification
            if len(buffer) >= WINDOW_SIZE:
                features = extract_features(buffer[-WINDOW_SIZE:])
                if features:
                    features_arr = np.array(features).reshape(1, -1)
                    probabilities = model.predict_proba(features_arr)[0]
                    confidence = np.max(probabilities)
                    predicted_word = model.predict(features_arr)[0]

                    now = time.time()
                    time_since_last = now - last_word_time

                    # Only show result if confident enough and not in cooldown
                    if confidence >= CONFIDENCE_MIN and time_since_last > COOLDOWN:
                        timestamp = datetime.now().strftime("%H:%M:%S")

                        # Big clear display
                        print(f"\n{'█'*40}")
                        print(f"  WORD DETECTED: {predicted_word.upper()}")
                        print(f"  Confidence:    {confidence*100:.1f}%")
                        print(f"  Time:          {timestamp}")
                        print(f"{'█'*40}\n")

                        last_word = predicted_word
                        last_word_time = now

                        # Clear buffer after detection
                        buffer = []
                    else:
                        # Slide window forward
                        buffer = buffer[STEP_SIZE:]

    except KeyboardInterrupt:
        print("\n\nStopped live recognition.")


# ─────────────────────────────────────────────
# SHOW TRAINING DATA SUMMARY
# ─────────────────────────────────────────────
def show_summary():
    if not os.path.isfile(DATA_FILE):
        print("\nNo training data recorded yet.")
        print(f"Record data with emg_recorder.py first.")
        return
    df = pd.read_csv(DATA_FILE)
    print(f"\n{'─'*40}")
    print(f"  TRAINING DATA SUMMARY")
    print(f"{'─'*40}")
    print(f"  Total recordings : {len(df)}")
    print(f"  Words            : {df['word'].nunique()}")
    print(f"\n  Per word:")
    counts = df["word"].value_counts()
    for word, count in counts.items():
        bar = "█" * (count // 5)
        status = "✓ READY" if count >= 50 else f"need {50-count} more"
        print(f"    {word:<12} {count:>3}  {bar}  {status}")
    model_status = "✓ exists" if os.path.isfile(MODEL_FILE) else "✗ not trained yet"
    print(f"\n  Trained model    : {model_status}")
    print(f"{'─'*40}\n")


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────
def main():
    print("\n" + "═"*50)
    print("  EMG WORD CLASSIFIER")
    print("═"*50)

    model = None
    words = None

    # Auto load model if it exists
    if os.path.isfile(MODEL_FILE):
        model, words = load_model()

    while True:
        print("\nOPTIONS:")
        print("  [1] Train model on recorded data")
        print("  [2] Start live recognition")
        print("  [3] Show training data summary")
        print("  [4] Quit")
        print()
        choice = input("Choose an option [1-4]: ").strip()

        if choice == "1":
            model, words = train_model()

        elif choice == "2":
            if model is None:
                print("\nNo model loaded. Train first using option 1.")
                continue
            print("\nConnecting to Arduino for live recognition...")
            ser = connect_arduino()
            try:
                live_recognition(ser, model, words)
            finally:
                ser.close()

        elif choice == "3":
            show_summary()

        elif choice == "4":
            print("\nExiting.")
            break

        else:
            print("Invalid choice. Enter 1-4.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
