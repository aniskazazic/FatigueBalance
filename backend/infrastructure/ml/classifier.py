# backend/infrastructure/ml/classifier.py - FIXED SA PRAVIM INCREMENTAL LEARNINGOM
import joblib
import numpy as np
import pandas as pd
import os
import time
from typing import List, Tuple, Optional, Dict, Any
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from infrastructure.ml.preprocessing import distance_to_km, normalize_position, normalize_activity
from infrastructure.ml.metrics_store import save_metrics, load_metrics

class FatigueClassifier:
    """ML klasa za predikciju fatigue score-a - SA PRAVIM INCREMENTAL LEARNING"""
    
    def __init__(self, model_file: str = "fatigue_model.joblib"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_file = os.path.join(base_dir, model_file) if not os.path.isabs(model_file) else model_file
        self.model: Optional[MLPRegressor] = None
        
        # Encoderi za kategoričke varijable
        self.position_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()
        
        # Definiši moguće vrijednosti
        self.positions = ["goalkeeper", "defender", "midfielder", "forward"]
        self.activities = ["practice", "game"]
        
        # Fit encodere
        self.position_encoder.fit(self.positions)
        self.activity_encoder.fit(self.activities)
        
        # Features metadata
        # NOTE: do not include `injury_illness` as an input feature for the
        # fatigue regressor by default to avoid potential data-leakage where
        # injury may be a consequence of fatigue. Injury is predicted via a
        # separate model (`injury_model`). If you intentionally want injury
        # as an input, enable it explicitly in training pipelines.
        self.feature_names = [
            'position', 'activity_type', 'sleep_hours', 'stress_level',
            'distance_km', 'sprint_count', 'soreness', 'rpe'
        ]
        self.n_features = len(self.feature_names)
        
        # Training history (ZA INCREMENTAL LEARNING)
        self.training_history = []
        self.initial_examples = []  # Čuva inicijalne primjere
        self.training_dataset_X = []
        self.training_dataset_y = []
        self.baseline_example_count = 16

        # Scaler for fatigue regression (saved alongside model when trained from CSV)
        self.scaler = None
        self.scaler_file = os.path.splitext(self.model_file)[0] + ".scaler.joblib"
        
        self._load_or_create_model()

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.injury_model_file = os.path.join(base_dir, "injury_model.joblib")
        self.injury_model = None
        self.injury_feature_columns = [
            "Sleep_Duration", "Stress", "Distance_km", "Soreness", "RPE"
        ]
        self.injury_metrics: Dict[str, Any] = {}
        self._load_or_train_injury_model()

    def _load_or_create_model(self):
        """Učitaj postojeći model ili kreiraj novi"""
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            # Attempt to load scaler if it exists (backwards compatible)
            try:
                if os.path.exists(self.scaler_file):
                    self.scaler = joblib.load(self.scaler_file)
                    if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.n_features:
                        print(f"⚠️ Scaler feature size mismatch: expected {self.n_features}, got {self.scaler.n_features_in_}. Ignoring scaler.")
                        self.scaler = None
                    else:
                        print(f"✓ Scaler učitan iz {self.scaler_file}")
            except Exception:
                self.scaler = None
            if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != self.n_features:
                print(f"⚠️ Loaded fatigue model expects {self.model.n_features_in_} features but current input has {self.n_features}. Reinitializing model.")
                self.model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=200,
                    random_state=42,
                    warm_start=True
                )
                self._initialize_model()
                joblib.dump(self.model, self.model_file)
                print(f"✓ Reinitialized fatigue model with current feature schema")
            else:
                print(f"✓ Model učitan iz {self.model_file}")

            # Čak i kada je učitan, inicijalizuj training_dataset za confidence calculation
            if not self.training_dataset_X or len(self.training_dataset_X) == 0:
                self._populate_training_dataset_from_examples()
        else:
            # Kreiraj novi model
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=200,
                random_state=42,
                warm_start=True  # OVO POMAŽE KOD RETRAIN-A
            )
            self._initialize_model()
            joblib.dump(self.model, self.model_file)
            print(f"✓ Model inicijaliziran")
    
    def _populate_training_dataset_from_examples(self):
        """Popuni training_dataset iz inicijalnih primjera za confidence calculation"""
        examples = [
            (0, 0, 8.0, 2, 5.0, 10, 2, 3, 25.0),
            (1, 0, 7.5, 3, 6.0, 15, 3, 4, 30.0),
            (2, 0, 8.0, 2, 7.0, 20, 2, 3, 28.0),
            (3, 0, 7.0, 3, 8.0, 25, 3, 4, 35.0),
            (3, 0, 6.0, 5, 8.0, 25, 5, 6, 50.0),
            (2, 1, 7.0, 4, 9.0, 30, 4, 5, 52.0),
            (1, 1, 6.5, 6, 7.5, 20, 5, 6, 48.0),
            (3, 1, 6.0, 5, 9.5, 32, 6, 6, 55.0),
            (3, 1, 5.0, 7, 10.0, 35, 7, 8, 70.0),
            (2, 1, 5.5, 8, 11.0, 40, 7, 8, 72.0),
            (1, 1, 6.0, 7, 8.5, 25, 6, 7, 65.0),
            (3, 1, 5.0, 8, 10.5, 38, 8, 8, 75.0),
            (3, 1, 4.0, 9, 12.0, 45, 9, 9, 88.0),
            (2, 1, 4.5, 9, 11.5, 42, 9, 10, 92.0),
            (1, 1, 5.0, 8, 9.0, 30, 8, 9, 82.0),
            (2, 1, 4.0, 10, 12.0, 45, 10, 10, 95.0),
        ]

        for pos, act, sleep, stress, dist, sprints, soreness, rpe, fatigue in examples:
            features = [pos, act, sleep, stress, dist, sprints, soreness, rpe]
            try:
                encoded = self._encode_features(features)
                self.training_dataset_X.append(np.array(encoded, dtype=float))
                self.training_dataset_y.append(float(fatigue))
            except Exception as e:
                print(f"⚠️ Upozorenje pri popunjavanju training dataset-a: {e}")
    
    def _initialize_model(self):
        """Inicijalizacija s osnovnim primjerima i sačuvaj ih"""
        X_init = []
        y_init = []
        
        # Format: [position, activity, sleep, stress, distance, sprints, soreness, rpe]
        examples = [
            # LOW fatigue (0-40)
            (0, 0, 8.0, 2, 5.0, 10, 2, 3, 25.0),
            (1, 0, 7.5, 3, 6.0, 15, 3, 4, 30.0),
            (2, 0, 8.0, 2, 7.0, 20, 2, 3, 28.0),
            (3, 0, 7.0, 3, 8.0, 25, 3, 4, 35.0),

            # MEDIUM fatigue (40-60)
            (3, 0, 6.0, 5, 8.0, 25, 5, 6, 50.0),
            (2, 1, 7.0, 4, 9.0, 30, 4, 5, 52.0),
            (1, 1, 6.5, 6, 7.5, 20, 5, 6, 48.0),
            (3, 1, 6.0, 5, 9.5, 32, 6, 6, 55.0),

            # HIGH fatigue (60-80)
            (3, 1, 5.0, 7, 10.0, 35, 7, 8, 70.0),
            (2, 1, 5.5, 8, 11.0, 40, 7, 8, 72.0),
            (1, 1, 6.0, 7, 8.5, 25, 6, 7, 65.0),
            (3, 1, 5.0, 8, 10.5, 38, 8, 8, 75.0),

            # CRITICAL fatigue (80-100)
            (3, 1, 4.0, 9, 12.0, 45, 9, 9, 88.0),
            (2, 1, 4.5, 9, 11.5, 42, 9, 10, 92.0),
            (1, 1, 5.0, 8, 9.0, 30, 8, 9, 82.0),
            (2, 1, 4.0, 10, 12.0, 45, 10, 10, 95.0),
        ]

        for pos, act, sleep, stress, dist, sprints, soreness, rpe, fatigue in examples:
            features = [pos, act, sleep, stress, dist, sprints, soreness, rpe]
            X_init.append(features)
            y_init.append(fatigue)
            
            # Sačuvaj inicijalne primjere za kasniji retrain
            self.initial_examples.append({
                'features': features,
                'fatigue_score': fatigue
            })
        
        self.model.fit(np.array(X_init), np.array(y_init))
        self.training_dataset_X = [np.array(x, dtype=float) for x in X_init]
        self.training_dataset_y = [float(y) for y in y_init]
        print(f"✓ Model inicijaliziran sa {len(X_init)} primjera")


    def _load_or_train_injury_model(self):
        if os.path.exists(self.injury_model_file):
            try:
                bundle = joblib.load(self.injury_model_file)
                self.injury_model = bundle.get("pipeline")
                self.injury_feature_columns = bundle.get(
                    "feature_columns", self.injury_feature_columns
                )
                self.injury_metrics = bundle.get("metrics", {})
                print(f"✓ Injury model učitan iz {self.injury_model_file}")
                return
            except Exception as exc:
                print(f"⚠️ Injury model load failed: {exc}")

        self._train_injury_model()

    def _train_injury_model(self, csv_path: Optional[str] = None) -> bool:
        """Logistička regresija (binarna) za vjerovatnoću povrede — train/test + StandardScaler."""
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        csv_path = csv_path or os.path.join(data_dir, "Workout_Routine_Dirty.csv")
        try:
            df = pd.read_csv(csv_path)
            required = ["Sleep_Duration", "Stress", "Distance", "Soreness", "RPE", "Injury_Illness"]
            if not all(col in df.columns for col in required):
                print("⚠️ CSV nema potrebne kolone za injury model")
                self.injury_model = None
                return False

            X = pd.DataFrame({
                "Sleep_Duration": pd.to_numeric(df["Sleep_Duration"], errors="coerce").fillna(7),
                "Stress": pd.to_numeric(df["Stress"], errors="coerce").fillna(5),
                "Distance_km": df["Distance"].apply(distance_to_km),
                "Soreness": pd.to_numeric(df["Soreness"], errors="coerce").fillna(5),
                "RPE": pd.to_numeric(df["RPE"], errors="coerce").fillna(5),
            })
            y = (df["Injury_Illness"].astype(str).str.strip().str.lower() == "yes").astype(int)

            if y.nunique() < 2:
                print("⚠️ Injury target ima samo jednu klasu — preskačem trening")
                self.injury_model = None
                return False

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )

            # Apply SMOTE when injury labels are imbalanced to improve recall on the
            # minority injury class while preserving a held-out evaluation split.
            resampled = False
            try:
                counts = y_train.value_counts()
                if counts.max() / max(counts.min(), 1) > 2.0:
                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                    resampled = True
                    print("⚡ Applied SMOTE to balance injury training data")
            except Exception:
                pass

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)) if y.nunique() > 1 else 0.0,
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "positive_rate": float(y.mean()),
                "smote_applied": resampled,
            }
            self.injury_model = pipeline
            self.injury_feature_columns = list(X.columns)
            self.injury_metrics = metrics

            joblib.dump({
                "pipeline": pipeline,
                "feature_columns": self.injury_feature_columns,
                "metrics": metrics,
            }, self.injury_model_file)

            all_metrics = load_metrics()
            all_metrics["injury_logistic_regression"] = metrics
            save_metrics(all_metrics)

            print(
                f"✅ Injury LR: acc={metrics['accuracy']:.2f}, "
                f"F1={metrics['f1']:.2f}, AUC={metrics['roc_auc']:.2f}"
            )
            return True
        except Exception as e:
            print(f"❌ Greška pri treniranju injury modela: {e}")
            self.injury_model = None
            return False

    def _injury_feature_row(self, features: List) -> pd.DataFrame:
        sleep = float(features[2])
        stress = float(features[3])
        distance_km = distance_to_km(features[4])
        soreness = float(features[6]) if len(features) > 6 else 5.0
        rpe = float(features[7]) if len(features) > 7 else 5.0
        row = {
            "Sleep_Duration": sleep,
            "Stress": stress,
            "Distance_km": distance_km,
            "Soreness": soreness,
            "RPE": rpe,
        }
        return pd.DataFrame([row], columns=self.injury_feature_columns)

    def predict_injury_prob(self, features: List) -> float:
        """Vjerovatnoća povrede (0–1) — logistička regresija sa fallback logikom."""
        if self.injury_model is None:
            # Fallback: izračunaj na osnovu features ako model nije dostupan
            return self._estimate_injury_from_features(features)
        
        try:
            X = self._injury_feature_row(features)
            prob = self.injury_model.predict_proba(X)[0][1]
            
            # Ako je predikcija preniska, koristi fallback logiku
            if prob < 0.15:
                fallback_prob = self._estimate_injury_from_features(features)
                # Koristi max od oba da izbegneš previsko nisko vrijednost
                prob = max(prob, fallback_prob * 0.5)
            
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as e:
            print(f"❌ Greška pri predikciji povrede: {e}")
            # Fallback na procjenu od features-a
            return self._estimate_injury_from_features(features)
    
    def _estimate_injury_from_features(self, features: List) -> float:
        """Procijeni vjerovatnoću povrede na osnovu features kada model nije dostupan."""
        try:
            sleep = float(features[2]) if len(features) > 2 else 7.0
            stress = float(features[3]) if len(features) > 3 else 5.0
            distance_km = float(features[4]) if len(features) > 4 else 0.0
            soreness = float(features[6]) if len(features) > 6 else 5.0
            rpe = float(features[7]) if len(features) > 7 else 5.0
            
            # Logika: više rizika ako je mal spavanja, visok stress, duža distanca, veća bol
            sleep_factor = max(0, (7.0 - sleep) / 7.0) * 0.3  # Manje spavanja = veći rizik
            stress_factor = (stress / 10.0) * 0.3  # Viši stress = veći rizik
            distance_factor = min(distance_km / 12.0, 1.0) * 0.2  # Duža distanca = veći rizik
            soreness_factor = (soreness / 10.0) * 0.2  # Veća bol = veći rizik
            
            injury_prob = sleep_factor + stress_factor + distance_factor + soreness_factor
            injury_prob = np.clip(injury_prob, 0.05, 0.95)  # Min 5%, Max 95%
            
            return float(injury_prob)
        except:
            return 0.20  # Default moderate risk
    
    def _encode_features(self, features: List) -> np.ndarray:
        """Enkoduj kategoričke varijable"""
        position_str = features[0]
        activity_str = features[1]
        
        try:
            pos_encoded = self.position_encoder.transform([position_str])[0]
        except Exception:
            pos_encoded = self.position_encoder.transform(["midfielder"])[0]

        try:
            act_encoded = self.activity_encoder.transform([activity_str])[0]
        except Exception:
            act_encoded = self.activity_encoder.transform(["practice"])[0]
        
        numeric_features = [
            float(features[2]),  # sleep_hours
            float(features[3]),  # stress_level
            float(features[4]),  # distance_km
            float(features[5])   # sprint_count
        ]
        
        soreness = float(features[6]) if len(features) > 6 else 5.0
        rpe = float(features[7]) if len(features) > 7 else 5.0

        # Injury is not part of the fatigue regressor inputs (avoid leakage)
        return np.array([
            pos_encoded,
            act_encoded,
            *numeric_features,
            soreness,
            rpe
        ])

    def _prepare_features(self, features: List) -> np.ndarray:
        """Encode features and apply scaler if available."""
        encoded = self._encode_features(features).reshape(1, -1)
        if self.scaler is not None:
            try:
                encoded = self.scaler.transform(encoded)
            except Exception:
                pass
        return encoded

    def _prepare_batch_features(self, X_batch: np.ndarray) -> np.ndarray:
        """Encode a batch of raw feature rows and scale if needed."""
        X_encoded = np.array([self._encode_features(row.tolist() if hasattr(row, 'tolist') else row) for row in X_batch])
        if self.scaler is not None:
            try:
                X_encoded = self.scaler.transform(X_encoded)
            except Exception:
                pass
        return X_encoded

    def predict(self, features: List) -> Tuple[float, float]:
        """Napravi predikciju fatigue score-a"""
        encoded_features = self._encode_features(features)
        x = self._prepare_features(features)
        
        fatigue_score = self.model.predict(x)[0]
        fatigue_score = max(0.0, min(100.0, fatigue_score))
        
        # Confidence estimated from local training neighborhood
        confidence = 0.60  # Default base confidence
        
        # Ako imamo dovoljno primjera, izračunaj na osnovu udaljenosti od susednih primjera
        if len(self.training_dataset_X) >= 3:
            X_data = np.vstack(self.training_dataset_X)
            dists = np.linalg.norm(X_data - encoded_features, axis=1)
            nearest = np.argsort(dists)[:5]
            neighbor_scores = np.array(self.training_dataset_y)[nearest]
            neighbor_std = float(np.std(neighbor_scores))
            neighbor_dist = float(np.mean(dists[nearest]))

            distance_conf = 1.0 - min(neighbor_dist / 20.0, 1.0)
            variance_conf = 1.0 - min(neighbor_std / 30.0, 1.0)
            confidence = float(np.clip(0.4 * distance_conf + 0.6 * variance_conf, 0.25, 0.95))
        else:
            # Čak i bez dovoljno primjera, koristi model uncertainty za bolju procjenu
            # Koristi model's hidden layer aktivacije za procjenu pouzdanosti
            try:
                # Za MLP modele, mogu koristiti aktivacije da procijenim uncertainty
                if hasattr(self.model, 'n_outputs_'):
                    # Generička procjena na osnovu inputa
                    input_norm = np.linalg.norm(encoded_features)
                    if input_norm > 0:
                        confidence = 0.60 + (0.15 * min(input_norm / 50.0, 1.0))
                    else:
                        confidence = 0.65
            except:
                confidence = 0.65
        
        return float(fatigue_score), float(confidence)
    
    def train_single(self, features: List, fatigue_score: float) -> bool:
        """
        🔥 OVO JE KLJUČNA METODA - INCREMENTAL LEARNING!
        Treniraj model sa feedback-om
        """
        # 1. Dodaj u training history
        self.training_history.append({
            'features': features,
            'fatigue_score': fatigue_score,
            'timestamp': time.time()
        })
        
        print(f"📝 Memorisan feedback: fatigue={fatigue_score:.2f}")
        print(f"   Ukupno memorisanih feedbacka: {len(self.training_history)}")
        
        # 2. Automatski retrain nakon 3 nova feedbacka
        if len(self.training_history) >= 3:
            success = self._retrain_on_all_examples()
            return success
        
        return True
    
    def _retrain_on_all_examples(self):
        """Retrain model na SVIM primjerima (inicijalni + feedback)"""
        print("\n🔄 Pokrećem retrain na svim primjerima...")
        
        all_X = []
        all_y = []
        
        # 1. Dodaj INICIJALNE primjere
        for example in self.initial_examples:
            encoded = self._encode_features(example['features'])
            all_X.append(encoded)
            all_y.append(example['fatigue_score'])
        
        # 2. Dodaj FEEDBACK primjere
        for item in self.training_history:
            encoded = self._encode_features(item['features'])
            all_X.append(encoded)
            all_y.append(item['fatigue_score'])
        
        # 3. Treniraj na SVI primjerima
        X_array = np.array(all_X)
        y_array = np.array(all_y)
        
        print(f"   Treniram na {len(y_array)} primjera:")
        print(f"   - Inicijalni: {len(self.initial_examples)}")
        print(f"   - Feedback: {len(self.training_history)}")
        
        # KLJUČNO: Koristi warm_start=True da model nastavi učiti
        X_train = X_array
        if self.scaler is not None:
            try:
                X_train = self.scaler.transform(X_array)
            except Exception:
                pass

        self.model.fit(X_train, y_array)
        self.training_dataset_X = [np.array(x, dtype=float) for x in X_array]
        self.training_dataset_y = [float(y) for y in y_array]
        
        # Sačuvaj model
        joblib.dump(self.model, self.model_file)
        
        print(f"✅ Retrain završen! Model sačuvan.")
        
        # 4. Provjeri da li je naučio - predikcije za neke primjere
        if len(self.training_history) > 0:
            latest = self.training_history[-1]
            features = latest['features']
            true_score = latest['fatigue_score']
            
            pred_score, confidence = self.predict(features)
            error = abs(pred_score - true_score)
            
            print(f"   Test za najnoviji feedback:")
            print(f"   - True: {true_score:.1f}, Pred: {pred_score:.1f}")
            print(f"   - Error: {error:.1f} (manje je bolje)")
        
        return True
    
    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """Treniraj model s batch-om podataka"""
        X_encoded = []
        for features in X_batch:
            encoded = self._encode_features(features.tolist() if hasattr(features, 'tolist') else features)
            X_encoded.append(encoded)
        
        X_encoded = np.array(X_encoded)
        if self.scaler is not None:
            try:
                X_encoded = self.scaler.transform(X_encoded)
            except Exception:
                pass
        
        # Treniraj model
        self.model.fit(X_encoded, y_batch)
        
        joblib.dump(self.model, self.model_file)
        print(f"✅ Model treniran na {len(y_batch)} primjera")
    
    def get_model_info(self):
        """Vrati informacije o modelu"""
        return {
            "model_type": "MLPRegressor (Neural Network)",
            "supports_incremental": True,
            "positions": self.positions,
            "activities": self.activities,
            "features": self.feature_names,
            "n_features": self.n_features,
            "model_file": self.model_file,
            "exists": os.path.exists(self.model_file),
            "initial_examples": len(self.initial_examples),
            "feedback_learned": len(self.training_history),
            "total_training_examples": len(self.initial_examples) + len(self.training_history),
            "injury_model": {
                "algorithm": "LogisticRegression (binary classification)",
                "model_file": self.injury_model_file,
                "loaded": self.injury_model is not None,
                "features": self.injury_feature_columns,
                "metrics": self.injury_metrics,
            },
        }
    
    def get_learning_stats(self):
        """Statistike učenja"""
        if not self.training_history:
            return {"message": "No feedback learned yet"}
        
        scores = [h['fatigue_score'] for h in self.training_history]
        
        return {
            "feedback_learned": len(self.training_history),
            "avg_fatigue_feedback": np.mean(scores),
            "min_fatigue_feedback": np.min(scores),
            "max_fatigue_feedback": np.max(scores),
            "recent_feedback": scores[-5:] if len(scores) >= 5 else scores
        }
    
    # ============================================================================
    # CSV TRAINING (ostaje isto kao prije)
    # ============================================================================
    
    def train_from_csv(self, csv_path: str = "data/Workout_Routine_Dirty.csv"):
        """Učitaj CSV, treniraj MLP (regresija umora) i injury LR; vrati MAE, RMSE, R²."""
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", csv_path)
            )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "Fatigue" in df.columns:
            target_col = "Fatigue"
        elif "FatigueScore" in df.columns:
            target_col = "FatigueScore"
        elif "ProxyFatigue" in df.columns:
            target_col = "ProxyFatigue"
        else:
            sleep = pd.to_numeric(df.get("Sleep_Duration", 7), errors="coerce").fillna(7)
            stress = pd.to_numeric(df.get("Stress", 5), errors="coerce").fillna(5)
            rpe = pd.to_numeric(df.get("RPE", 5), errors="coerce").fillna(5)
            soreness = pd.to_numeric(df.get("Soreness", 5), errors="coerce").fillna(5)
            proxy = ((10 - sleep) * 5) + (stress * 5) + (rpe * 3) + (soreness * 2)
            df["ProxyFatigue"] = proxy.clip(0, 100)
            target_col = "ProxyFatigue"

        rows = []
        targets = []
        for _, row in df.iterrows():
            position = normalize_position(row.get("Position", "midfielder"))
            activity = normalize_activity(row.get("Session_Type", row.get("Activity", "practice")))
            sleep = float(row.get("Sleep_Duration", 7))
            stress = float(row.get("Stress", 5))
            distance = distance_to_km(row.get("Distance", 5))
            sprint = int(row.get("SprintCount", row.get("Acceleration_Count", 10)) or 10)
            soreness = float(row.get("Soreness", 5))
            rpe = float(row.get("RPE", 5))
            injury = 1 if str(row.get("Injury_Illness", "No")).strip().lower() in ("yes", "1", "true") else 0
            # Note: injury is computed for downstream injury model but is not
            # included as an input feature to the fatigue regressor to avoid
            # leakage (injury may be a consequence of fatigue).
            rows.append([position, activity, sleep, stress, distance, sprint, soreness, rpe])
            targets.append(float(row.get(target_col, 0)))

        X_encoded = np.array([self._encode_features(f) for f in rows])
        y = np.array(targets)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        # Fit scaler on training data and transform both sets to avoid leakage
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler
            # Save scaler for future incremental usage
            try:
                joblib.dump(scaler, self.scaler_file)
            except Exception:
                pass
        except Exception:
            # If scaling fails, fall back to unscaled data
            pass

        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=300,
            random_state=42,
            warm_start=True,
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        joblib.dump(self.model, self.model_file)
        self.training_dataset_X = [np.array(x, dtype=float) for x in X_encoded]
        self.training_dataset_y = [float(v) for v in y]

        fatigue_metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "target_column": target_col,
        }
        all_metrics = load_metrics()
        all_metrics["fatigue_mlp_regressor"] = fatigue_metrics
        save_metrics(all_metrics)

        self._train_injury_model(csv_path)
        return mae, rmse, r2