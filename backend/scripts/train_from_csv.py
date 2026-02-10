#!/usr/bin/env python
# train_from_csv.py - Pokreni ovaj script da treniraš model sa CSV podacima
import sys
sys.path.insert(0, 'backend')

from infrastructure.ml.classifier import FatigueClassifier

if __name__ == "__main__":
    print("\n🔥 Pokretanje treninga sa podacima iz CSV-a...")
    print("📁 CSV: data/Workout_Routine_Dirty.csv")
    print()
    
    # Kreiraj classifier i treniraj
    classifier = FatigueClassifier()
    mae, rmse, r2 = classifier.train_from_csv("data/Workout_Routine_Dirty.csv")
    
    print("\n" + "="*70)
    print("🎉 USPJEŠNO ZAVRŠENO!")
    print("="*70)
    print(f"✅ Model treniran i sačuvan u: fatigue_model.joblib")
    print(f"📊 Performanse:")
    print(f"   - MAE:  {mae:.2f}")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - R²:   {r2:.3f}")
    print()
    print("🚀 Sledeći korak:")
    print("   python main.py  - Pokreni agenta sa novim modelom")
    print("="*70)