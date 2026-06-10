import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infrastructure.ml.classifier import FatigueClassifier
from infrastructure.ml.risk_classifier import RiskClassifier

if __name__ == '__main__':
    fc = FatigueClassifier()
    rc = RiskClassifier(auto_train=False)

    features = ['midfielder', 'practice', 7.0, 5.0, 8.0, 20, 4.0, 5.0, 0]
    fatigue, confidence = fc.predict(features)
    print(f'fatigue={fatigue:.2f}, confidence={confidence:.2f}')
    print('risk model loaded:', rc.model is not None)

    if rc.model is not None:
        try:
            import pandas as pd
            X = pd.DataFrame([[7.0, 5.0, 8.0, 4.0, 5.0]], columns=['Sleep_Duration', 'Stress', 'Distance_km', 'Soreness', 'RPE'])
            pred = rc.model.predict(X)
            labels = {0: 'low', 1: 'medium', 2: 'high', 3: 'critical'}
            print('risk prediction:', labels.get(int(pred[0]), pred[0]))
        except Exception as e:
            print('risk prediction error:', e)
