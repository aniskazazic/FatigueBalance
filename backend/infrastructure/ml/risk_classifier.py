import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from domain.entities import RiskLevel
from infrastructure.database import get_connection
from infrastructure.ml.preprocessing import distance_to_km
from infrastructure.ml.metrics_store import save_metrics, load_metrics


class RiskClassifier:
    """Logistic regression (multinomial) for LOW/MEDIUM/HIGH/CRITICAL risk levels."""

    def __init__(self, model_file: str = "risk_model.joblib", auto_train: bool = True):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_file = os.path.join(base_dir, model_file) if not os.path.isabs(model_file) else model_file
        self.model: Optional[Pipeline] = None
        self.feature_cols = [
            "Sleep_Duration",
            "Stress",
            "Distance_km",
            "Soreness",
            "RPE",
        ]
        self.risk_metrics: Dict[str, Any] = {}
        self.trained_columns: Optional[list] = None
        self.feedback_examples: List[Tuple[List[float], int]] = []
        self._load_or_create()

        if self.model is None and auto_train:
            trained_from_db = self.train_from_db()
            if not trained_from_db:
                self.train_from_csv()

    def _load_or_create(self):
        if os.path.exists(self.model_file):
            try:
                bundle = joblib.load(self.model_file)
                if isinstance(bundle, dict) and "pipeline" in bundle:
                    self.model = bundle["pipeline"]
                    self.trained_columns = bundle.get("feature_columns", self.feature_cols)
                    self.risk_metrics = bundle.get("metrics", {})
                else:
                    self.model = None
                    print("[WARN] Old risk model format - will retrain")
                    return
                print(f"[OK] Risk model loaded from {self.model_file}")
            except Exception as e:
                print(f"[WARN] Failed to load risk model: {e}")
                self.model = None
        else:
            self.model = None

    def train_from_csv(self, csv_path: str = "data/Workout_Routine_Dirty.csv") -> bool:
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", csv_path))
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV not found: {csv_path}")
            return False

        df = pd.read_csv(csv_path)

        if "RiskLevel" in df.columns:
            y_raw = df["RiskLevel"].astype(str).str.lower()
            mapping = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            y = y_raw.map(mapping).fillna(0).astype(int)
        elif "Fatigue" in df.columns or "FatigueScore" in df.columns or "ProxyFatigue" in df.columns:
            if "Fatigue" in df.columns:
                col = "Fatigue"
            elif "FatigueScore" in df.columns:
                col = "FatigueScore"
            else:
                col = "ProxyFatigue"
            fatigue = pd.to_numeric(df[col], errors="coerce").fillna(0)
            y = pd.cut(fatigue, bins=[-1, 40, 60, 80, 100], labels=[0, 1, 2, 3]).astype(int)
        else:
            print("⚠️ CSV missing RiskLevel or Fatigue columns - cannot create labels")
            return False

        if "Sleep_Duration" not in df.columns:
            print("⚠️ No available feature columns for risk training")
            return False

        X = pd.DataFrame({
            "Sleep_Duration": pd.to_numeric(df["Sleep_Duration"], errors="coerce").fillna(7),
            "Stress": pd.to_numeric(df["Stress"], errors="coerce").fillna(5),
            "Distance_km": df["Distance"].apply(distance_to_km) if "Distance" in df.columns else 5.0,
            "Soreness": pd.to_numeric(df["Soreness"], errors="coerce").fillna(5),
            "RPE": pd.to_numeric(df["RPE"], errors="coerce").fillna(5),
        })
        return self._train_model(X, y)

    def train_from_db(self, min_examples: int = 16) -> bool:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SleepHours, StressLevel, DistanceKm, Soreness, RPE,
                       RiskLevel, FatigueScore
                FROM TrainingSessions
                WHERE Status = 'processed'
                  AND (RiskLevel IS NOT NULL OR FatigueScore IS NOT NULL)
            """)
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"⚠️ RiskClassifier DB training unavailable: {e}")
            return False

        if not rows:
            return False

        X_rows = []
        y_rows = []
        for row in rows:
            sleep, stress, distance, soreness, rpe, risk_raw, fatigue_score = row
            soreness_val = float(soreness) if soreness is not None else 5.0
            rpe_val = float(rpe) if rpe is not None else 5.0
            label = self._parse_risk_label(risk_raw)
            if label is None and fatigue_score is not None:
                label = self._fatigue_score_to_risk(float(fatigue_score))
            if label is None:
                continue
            X_rows.append([
                float(sleep), float(stress), distance_to_km(distance), soreness_val, rpe_val
            ])
            y_rows.append(label)

        if len(y_rows) < min_examples:
            print(f"⚠️ Not enough DB risk examples ({len(y_rows)}) to train")
            return False

        X = pd.DataFrame(X_rows, columns=self.feature_cols)
        y = pd.Series(y_rows, dtype=int)
        return self._train_model(X, y)

    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        if X.empty or y.empty or len(y) < 4:
            print("⚠️ Risk classifier training skipped due to insufficient data")
            return False

        X_clean = X.fillna(X.median()).astype(float)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = X_clean, X_clean, y, y

        # Optionally apply SMOTE if training data is highly imbalanced
        smote_applied = False
        try:
            counts = y_train.value_counts()
            if counts.max() / max(counts.min(), 1) > 3.0:
                try:
                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE(random_state=42)
                    X_res, y_res = sm.fit_resample(X_train, y_train)
                    X_train, y_train = X_res, y_res
                    smote_applied = True
                    print("⚡ Applied SMOTE to balance risk training data")
                except Exception:
                    pass
        except Exception:
            pass

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
        ])
        pipeline.fit(X_train, y_train)

        if len(X_test) > 0:
            preds = pipeline.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            balanced_acc = float(balanced_accuracy_score(y_test, preds))
            macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        else:
            acc = 1.0
            balanced_acc = 1.0
            macro_f1 = 1.0

        self.model = pipeline
        self.trained_columns = list(X.columns)
        self.risk_metrics = {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "smote_applied": smote_applied,
        }
        try:
            joblib.dump({
                "pipeline": pipeline,
                "feature_columns": self.trained_columns,
                "metrics": self.risk_metrics,
            }, self.model_file)
        except Exception as e:
            print(f"⚠️ Unable to save risk model: {e}")

        all_metrics = load_metrics()
        all_metrics["risk_logistic_regression"] = self.risk_metrics
        save_metrics(all_metrics)

        print(f"✅ Risk LR: accuracy={acc:.2f}, balanced_accuracy={balanced_acc:.2f}, macro-F1={macro_f1:.2f}, smote_applied={smote_applied}")
        print(f"   Features used: {self.trained_columns}")
        return True

    def add_feedback_example(self, features: List, user_label: str) -> bool:
        risk_label = self._parse_risk_label(user_label)
        if risk_label is None:
            return False

        try:
            sleep = float(features[2])
            stress = float(features[3])
            distance = distance_to_km(features[4])
            soreness = float(features[6]) if len(features) > 6 else 5.0
            rpe = float(features[7]) if len(features) > 7 else 5.0
        except Exception:
            return False

        self.feedback_examples.append(([sleep, stress, distance, soreness, rpe], risk_label))
        return True

    def retrain_on_feedback(self) -> bool:
        if not self.feedback_examples:
            return False

        X_rows = [example[0] for example in self.feedback_examples]
        y_rows = [example[1] for example in self.feedback_examples]
        X = pd.DataFrame(X_rows, columns=self.feature_cols)
        y = pd.Series(y_rows, dtype=int)

        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SleepHours, StressLevel, DistanceKm, Soreness, RPE, RiskLevel, FatigueScore
                FROM TrainingSessions
                WHERE Status = 'processed'
                  AND (RiskLevel IS NOT NULL OR FatigueScore IS NOT NULL)
            """)
            rows = cursor.fetchall()
            conn.close()
        except Exception:
            rows = []

        db_rows = []
        db_y = []
        for row in rows:
            sleep, stress, distance, soreness, rpe, risk_raw, fatigue_score = row
            label = self._parse_risk_label(risk_raw)
            if label is None and fatigue_score is not None:
                label = self._fatigue_score_to_risk(float(fatigue_score))
            if label is None:
                continue
            db_rows.append([float(sleep), float(stress), float(distance), float(soreness) if soreness is not None else 5.0, float(rpe) if rpe is not None else 5.0])
            db_y.append(label)

        if db_rows:
            X = pd.concat([X, pd.DataFrame(db_rows, columns=self.feature_cols)], ignore_index=True)
            y = pd.concat([y, pd.Series(db_y, dtype=int)], ignore_index=True)

        return self._train_model(X, y)

    def _parse_risk_label(self, raw_label) -> Optional[int]:
        if raw_label is None:
            return None

        label = str(raw_label).strip().lower()
        if label in {"low", "medium", "high", "critical"}:
            mapping = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            return mapping[label]

        action_map = {
            "cleared": 0,
            "monitor": 1,
            "reduce_intensity": 2,
            "rest_recommended": 2,
            "must_rest": 3
        }
        if label in action_map:
            return action_map[label]

        try:
            score = float(label)
            return self._fatigue_score_to_risk(score)
        except Exception:
            return None

    def _fatigue_score_to_risk(self, score: float) -> int:
        if score < 40:
            return 0
        if score < 60:
            return 1
        if score < 80:
            return 2
        return 3

    def predict_risk_level(self, features: List) -> RiskLevel:
        """Predict risk level from a feature list in the same order as TrainingSession.extract_features().

        If model is not available, fall back to simple thresholds on fatigue if provided.
        """
        # features: [position, activity, sleep, stress, distance, sprint_count, soreness, rpe, injury]
        try:
            sleep = float(features[2])
            stress = float(features[3])
            distance = distance_to_km(features[4])
            soreness = float(features[6]) if len(features) > 6 else 5.0
            rpe = float(features[7]) if len(features) > 7 else 5.0
        except Exception:
            return RiskLevel.LOW

        X = np.array([[sleep, stress, distance, soreness, rpe]])
        if self.model is not None:
            columns = self.trained_columns if self.trained_columns is not None else self.feature_cols
            X = pd.DataFrame(X, columns=columns)

        if self.model is None:
            # fallback simple rule
            # compute a fatigue-like proxy
            score = ( (10 - sleep) * 5 ) + (stress * 5) + (rpe * 3) + (soreness * 2)
            if score < 40:
                return RiskLevel.LOW
            if score < 60:
                return RiskLevel.MEDIUM
            if score < 80:
                return RiskLevel.HIGH
            return RiskLevel.CRITICAL

        pred = self.model.predict(X)[0]
        mapping = {0: RiskLevel.LOW, 1: RiskLevel.MEDIUM, 2: RiskLevel.HIGH, 3: RiskLevel.CRITICAL}
        return mapping.get(int(pred), RiskLevel.LOW)

    def get_feature_importance(self):
        if self.model is None:
            return {}
        try:
            clf = self.model.named_steps["clf"]
            coef = np.abs(clf.coef_).mean(axis=0)
            cols = self.trained_columns or self.feature_cols
            return dict(zip(cols, coef.tolist()))
        except Exception:
            return {}

    def get_model_info(self):
        return {
            "model_type": "LogisticRegression (multinomial)",
            "model_file": self.model_file,
            "trained_columns": self.trained_columns,
            "model_exists": self.model is not None,
            "feedback_examples": len(self.feedback_examples),
            "metrics": self.risk_metrics,
            "feature_importance": self.get_feature_importance(),
        }