"""Train RiskClassifier and FatigueClassifier from CSV (if available).

Usage:
    python scripts/train_models.py [path/to/Workout_Routine_Dirty.csv]
"""
import sys
import os

# Ensure the parent directory is on sys.path so package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.ml.risk_classifier import RiskClassifier
from infrastructure.ml.classifier import FatigueClassifier
from infrastructure.ml.preprocessing import normalize_position, normalize_activity, distance_to_km


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "data", "Workout_Routine_Dirty.csv")
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path} — nothing to train.")
        return

    # Load CSV first so we can create proxy targets if needed and pass
    # a prepared CSV to the RiskClassifier.
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Create proxy fatigue if missing so RiskClassifier can be trained
    if "Fatigue" not in df.columns and "FatigueScore" not in df.columns:
        print("No explicit fatigue column found — creating proxy fatigue target for risk training.")
        df['Sleep_Duration'] = pd.to_numeric(df.get('Sleep_Duration', 7)).fillna(7)
        df['Stress'] = pd.to_numeric(df.get('Stress', 5)).fillna(5)
        df['RPE'] = pd.to_numeric(df.get('RPE', 5)).fillna(5)
        df['Soreness'] = pd.to_numeric(df.get('Soreness', 5)).fillna(5)
        proxy = ((10 - df['Sleep_Duration']) * 5) + (df['Stress'] * 5) + (df['RPE'] * 3) + (df['Soreness'] * 2)
        df['ProxyFatigue'] = proxy.clip(lower=0, upper=100)
        # save a temporary CSV for risk training
        tmp_path = os.path.join(base_dir, "data", "Workout_Routine_Dirty_with_proxy.csv")
        df.to_csv(tmp_path, index=False)
        rc_csv_path = tmp_path
    else:
        rc_csv_path = csv_path

    print("Training RiskClassifier...")
    rc = RiskClassifier()
    rc.train_from_csv(rc_csv_path)

    print("Attempting to train FatigueClassifier from CSV...")
    fc = FatigueClassifier()
    fc.train_from_csv(csv_path)
    print("Fatigue model training complete and saved.")


if __name__ == '__main__':
    main()
