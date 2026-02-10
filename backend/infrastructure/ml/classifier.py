# backend/infrastructure/ml/classifier.py - FIXED SA PRAVIM INCREMENTAL LEARNINGOM
import joblib
import numpy as np
import pandas as pd
import os
import time
from typing import List, Tuple, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class FatigueClassifier:
    """ML klasa za predikciju fatigue score-a - SA PRAVIM INCREMENTAL LEARNING"""
    
    def __init__(self, model_file: str = "fatigue_model.joblib"):
        self.model_file = model_file
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
        self.feature_names = [
            'position', 'activity_type', 'sleep_hours', 'stress_level',
            'distance_km', 'sprint_count', 'soreness', 'rpe', 'injury_illness'
        ]
        self.n_features = len(self.feature_names)
        
        # Training history (ZA INCREMENTAL LEARNING)
        self.training_history = []
        self.initial_examples = []  # Čuva inicijalne primjere
        
        self._load_or_create_model()

    def _load_or_create_model(self):
        """Učitaj postojeći model ili kreiraj novi"""
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            print(f"✓ Model učitan iz {self.model_file}")
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
    
    def _initialize_model(self):
        """Inicijalizacija s osnovnim primjerima i sačuvaj ih"""
        X_init = []
        y_init = []
        
        # Format: [position, activity, sleep, stress, distance, sprints, soreness, rpe, injury]
        examples = [
            # LOW fatigue (0-40)
            (0, 0, 8.0, 2, 5.0, 10, 2, 3, 0, 25.0),
            (1, 0, 7.5, 3, 6.0, 15, 3, 4, 0, 30.0),
            (2, 0, 8.0, 2, 7.0, 20, 2, 3, 0, 28.0),
            (3, 0, 7.0, 3, 8.0, 25, 3, 4, 0, 35.0),
            
            # MEDIUM fatigue (40-60)
            (3, 0, 6.0, 5, 8.0, 25, 5, 6, 0, 50.0),
            (2, 1, 7.0, 4, 9.0, 30, 4, 5, 0, 52.0),
            (1, 1, 6.5, 6, 7.5, 20, 5, 6, 0, 48.0),
            (3, 1, 6.0, 5, 9.5, 32, 6, 6, 0, 55.0),
            
            # HIGH fatigue (60-80)
            (3, 1, 5.0, 7, 10.0, 35, 7, 8, 0, 70.0),
            (2, 1, 5.5, 8, 11.0, 40, 7, 8, 0, 72.0),
            (1, 1, 6.0, 7, 8.5, 25, 6, 7, 0, 65.0),
            (3, 1, 5.0, 8, 10.5, 38, 8, 8, 0, 75.0),
            
            # CRITICAL fatigue (80-100)
            (3, 1, 4.0, 9, 12.0, 45, 9, 9, 1, 88.0),
            (2, 1, 4.5, 9, 11.5, 42, 9, 10, 1, 92.0),
            (1, 1, 5.0, 8, 9.0, 30, 8, 9, 0, 82.0),
            (2, 1, 4.0, 10, 12.0, 45, 10, 10, 1, 95.0),
        ]
        
        for pos, act, sleep, stress, dist, sprints, soreness, rpe, injury, fatigue in examples:
            features = [pos, act, sleep, stress, dist, sprints, soreness, rpe, injury]
            X_init.append(features)
            y_init.append(fatigue)
            
            # Sačuvaj inicijalne primjere za kasniji retrain
            self.initial_examples.append({
                'features': features,
                'fatigue_score': fatigue
            })
        
        self.model.fit(np.array(X_init), np.array(y_init))
        print(f"✓ Model inicijaliziran sa {len(X_init)} primjera")
    
    def _encode_features(self, features: List) -> np.ndarray:
        """Enkoduj kategoričke varijable"""
        position_str = features[0]
        activity_str = features[1]
        
        pos_encoded = self.position_encoder.transform([position_str])[0]
        act_encoded = self.activity_encoder.transform([activity_str])[0]
        
        numeric_features = [
            float(features[2]),  # sleep_hours
            float(features[3]),  # stress_level
            float(features[4]),  # distance_km
            float(features[5])   # sprint_count
        ]
        
        soreness = float(features[6]) if len(features) > 6 else 5.0
        rpe = float(features[7]) if len(features) > 7 else 5.0
        injury = float(features[8]) if len(features) > 8 else 0.0
        
        return np.array([
            pos_encoded,
            act_encoded,
            *numeric_features,
            soreness,
            rpe,
            injury
        ])
    
    def predict(self, features: List) -> Tuple[float, float]:
        """Napravi predikciju fatigue score-a"""
        encoded_features = self._encode_features(features)
        x = encoded_features.reshape(1, -1)
        
        fatigue_score = self.model.predict(x)[0]
        fatigue_score = max(0.0, min(100.0, fatigue_score))
        
        # Confidence based na broju naučenih primjera
        total_examples = len(self.initial_examples) + len(self.training_history)
        confidence = min(0.9, 0.5 + (total_examples * 0.01))
        
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
        self.model.fit(X_array, y_array)
        
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
            encoded = self._encode_features(features.tolist())
            X_encoded.append(encoded)
        
        X_encoded = np.array(X_encoded)
        
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
            "total_training_examples": len(self.initial_examples) + len(self.training_history)
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
        """Učitaj CSV, pripremi podatke i treniraj model"""
        # OVAJ DIO OSTAJE ISTI KAO U TVOM ORIGINALNOM KODU
        # (samo kopiraj iz svog postojećeg fajla)
        pass