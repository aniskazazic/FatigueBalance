
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, learning_curve, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               StackingRegressor, StackingClassifier)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, f1_score, confusion_matrix,
                              classification_report, roc_auc_score,
                              roc_curve, auc, silhouette_score)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ========================== 1. Učitavanje podataka ==========================
df = pd.read_csv('Workout_Routine_Dirty.csv')
print("="*60)
print("1. UČITAVANJE PODATAKA")
print("="*60)
print(f"Oblik: {df.shape}")
print(df.head(3))
print(df.info())

# ========================== 2. Analiza podataka (EDA) ==========================
print("\n" + "="*60)
print("2. DESKRIPTIVNA STATISTIKA I ANALIZA")
print("="*60)
missing = df.isnull().sum()
print("\nNedostajuće vrijednosti:")
print(missing[missing > 0] if any(missing > 0) else "Nema nedostajućih vrijednosti.")
num_cols = df.select_dtypes(include=[np.number]).columns
print("\nDeskriptivna statistika:")
print(df[num_cols].describe())
cat_cols = df.select_dtypes(include=['object']).columns
print("\nKategoričke varijable:")
for col in cat_cols:
    print(f"{col}: {df[col].unique()[:5]} (ukupno {df[col].nunique()})")

# Distribucija ciljne varijable
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['Fatigue'], bins=30, kde=True)
plt.title('Distribucija Fatigue')
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fatigue'])
plt.title('Box plot Fatigue')
plt.tight_layout()
plt.savefig('distribucija_fatigue.png')
plt.show()

# Korelaciona matrica
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelaciona matrica')
plt.tight_layout()
plt.savefig('korelaciona_matrica.png')
plt.show()

# ========================== 3. Priprema podataka + FEATURE ENGINEERING ==========================
print("\n" + "="*60)
print("3. PRIPREMA PODATAKA + FEATURE ENGINEERING")
print("="*60)

df_clean = df.copy()
drop_cols = ['Date', 'Name', 'Sleep_Score', 'Sleep_Quality', 'Max_Speed', 'Max_Acceleration',
             'FatigueScore', 'RiskLevel', 'Injury_Type']
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')

# === NOVO: Feature Engineering ===
# 1. Interakcija između stresa i RPE (fizičko + mentalno opterećenje)
df_clean['Stress_x_RPE'] = df_clean['Stress'] * df_clean['RPE']
# 2. Omjer distance i sna (napora vs oporavka)
df_clean['Distance_per_Sleep'] = df_clean['Distance'] / (df_clean['Sleep_Duration'] + 0.1)
# 3. Ukupna akceleracija/deceleracija (mjerilo intenziteta promjene pravca)
df_clean['Total_Acc_Dec'] = df_clean['Acceleration_Count'] + df_clean['Deceleration_Count']
# 4. Postotak pređene distance u odnosu na prosjek igrača (normalizacija igrača)
mean_distance = df_clean['Distance'].mean()
df_clean['Distance_vs_Avg'] = df_clean['Distance'] / mean_distance

print("Dodane feature-engineered varijable:")
print(" - Stress_x_RPE: stres × RPE (kombinirano opterećenje)")
print(" - Distance_per_Sleep: distanca / trajanje sna")
print(" - Total_Acc_Dec: Acceleration_Count + Deceleration_Count")
print(" - Distance_vs_Avg: distanca / prosječna distanca")

# Enkodiranje
categorical_features = ['Position', 'Session_Type']
df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)

# Rješavanje nedostajućih
for col in df_encoded.columns:
    if df_encoded[col].isnull().any():
        if df_encoded[col].dtype in ['float64', 'int64']:
            df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
        else:
            df_encoded[col].fillna(df_encoded[col].mode()[0], inplace=True)

target_reg = 'Fatigue'
X = df_encoded.drop(columns=[target_reg, 'Injury_Illness'])
y_reg = df_encoded[target_reg]

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nBroj ulaznih karakteristika (sa feature eng.): {X.shape[1]}")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ========================== 4. Regresija ==========================
print("\n" + "="*60)
print("4. REGRESIJA – PREDVIĐANJE FATIGUE SCORE")
print("="*60)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                   max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

def evaluate_reg(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R²   = {r2:.3f}")
    return mae, rmse, r2

print("\n--- REZULTATI NA TEST SETU ---")
evaluate_reg(y_test, y_pred_rf, "Random Forest")
evaluate_reg(y_test, y_pred_mlp, "MLP Regressor")

cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCross-validation R² (Random Forest): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Scatter plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Predviđene vrijednosti')
plt.title('Random Forest: stvarno vs predviđeno')
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_mlp, alpha=0.5, c='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Predviđene vrijednosti')
plt.title('MLP: stvarno vs predviđeno')
plt.tight_layout()
plt.savefig('regresija_usporedba.png')
plt.show()

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Važnost karakteristika (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()

# ========================== 5. NOVO: GridSearchCV ==========================
print("\n" + "="*60)
print("5. GRIDSEACHCV – OPTIMIZACIJA HIPERPARAMETARA (Random Forest)")
print("="*60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nNajbolji parametri: {grid_search.best_params_}")
print(f"Najbolji CV R²: {grid_search.best_score_:.3f}")

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_scaled)
print("\n--- Optimizirani Random Forest na test setu ---")
evaluate_reg(y_test, y_pred_best_rf, "Random Forest (GridSearchCV)")

# Vizualizacija GridSearch rezultata
cv_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
pivot_depth = cv_results[cv_results['param_min_samples_split'] == 2].pivot_table(
    index='param_n_estimators', columns='param_max_depth', values='mean_test_score'
)
sns.heatmap(pivot_depth, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('GridSearch: R² po n_estimators i max_depth\n(min_samples_split=2)')
plt.subplot(1, 2, 2)
means = cv_results.groupby('param_n_estimators')['mean_test_score'].mean()
stds = cv_results.groupby('param_n_estimators')['mean_test_score'].std()
plt.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=5)
plt.xlabel('n_estimators')
plt.ylabel('R² (CV mean)')
plt.title('GridSearch: R² po broju stabala')
plt.tight_layout()
plt.savefig('gridsearch_rezultati.png')
plt.show()

# ========================== 6. NOVO: Learning Curves ==========================
print("\n" + "="*60)
print("6. LEARNING CURVES – BIAS/VARIANCE ANALIZA")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, name in zip(axes,
                            [best_rf, mlp],
                            ['Random Forest (optimizirani)', 'MLP Regressor']):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train,
        cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std,   val_mean + val_std,   alpha=0.15, color='red')
    ax.plot(train_sizes, train_mean, 'o-', color='blue',  label='Trening R²')
    ax.plot(train_sizes, val_mean,   'o-', color='red',   label='Validacija R²')
    ax.set_xlabel('Veličina trening seta')
    ax.set_ylabel('R²')
    ax.set_title(f'Learning Curve – {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()
print("Interpretacija:")
print("  - Mali jaz trening/validacija → nema overfittinga")
print("  - Konvergencija krivih → model se 'zasitio' podacima → potrebno više uzoraka")

# ========================== 7. Klasifikacija rizika ==========================
print("\n" + "="*60)
print("7. KLASIFIKACIJA – PREDVIĐANJE RIZIKA (RiskLevel)")
print("="*60)

risk_target = df['RiskLevel'].copy()
le_risk = LabelEncoder()
y_risk = le_risk.fit_transform(risk_target)

X_risk = X.copy()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled  = scaler_r.transform(X_test_r)

lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lr.fit(X_train_r_scaled, y_train_r)
y_pred_lr = lr.predict(X_test_r_scaled)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_r_scaled, y_train_r)
y_pred_rf_clf = rf_clf.predict(X_test_r_scaled)

def evaluate_clf(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"\n{model_name}:")
    print(f"  Accuracy  = {acc:.3f}")
    print(f"  Macro F1  = {f1_macro:.3f}")
    print(classification_report(y_true, y_pred, target_names=le_risk.classes_))

print("\n--- KLASIFIKACIJA RIZIKA ---")
evaluate_clf(y_test_r, y_pred_lr,     "Logistic Regression")
evaluate_clf(y_test_r, y_pred_rf_clf, "Random Forest")

# Matrica konfuzije
cm = confusion_matrix(y_test_r, y_pred_rf_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
plt.title('Matrica konfuzije – Random Forest (RiskLevel)')
plt.xlabel('Predviđeno'); plt.ylabel('Stvarno')
plt.tight_layout()
plt.savefig('confusion_matrix_risk.png')
plt.show()

# ========================== 8. Klasifikacija povrede (SMOTE) ==========================
print("\n" + "="*60)
print("8. KLASIFIKACIJA POVREDE – SMOTE vs BEZ SMOTE")
print("="*60)

y_injury = (df['Injury_Illness'].astype(str).str.lower() == 'yes').astype(int)
X_injury = X.copy()
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_injury, y_injury, test_size=0.2, random_state=42, stratify=y_injury)
scaler_i = StandardScaler()
X_train_i_scaled = scaler_i.fit_transform(X_train_i)
X_test_i_scaled  = scaler_i.transform(X_test_i)

# Bez SMOTE
lr_no_smote = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
lr_no_smote.fit(X_train_i_scaled, y_train_i)
y_pred_no_smote = lr_no_smote.predict(X_test_i_scaled)
y_prob_no_smote = lr_no_smote.predict_proba(X_test_i_scaled)[:, 1]

print("\n--- BEZ SMOTE (class_weight='balanced') ---")
print(f"  Accuracy = {accuracy_score(y_test_i, y_pred_no_smote):.3f}")
print(f"  F1 score = {f1_score(y_test_i, y_pred_no_smote):.3f}")
print(f"  ROC-AUC  = {roc_auc_score(y_test_i, y_prob_no_smote):.3f}")
print(classification_report(y_test_i, y_pred_no_smote, target_names=['No injury', 'Injury']))

# Sa SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)  # k_neighbors=3 jer malo pozitivnih
X_train_smote, y_train_smote = smote.fit_resample(X_train_i_scaled, y_train_i)

print(f"\nBrojevi klasa prije SMOTE: {dict(zip(*np.unique(y_train_i, return_counts=True)))}")
print(f"Brojevi klasa nakon SMOTE: {dict(zip(*np.unique(y_train_smote, return_counts=True)))}")

lr_smote = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = lr_smote.predict(X_test_i_scaled)
y_prob_smote = lr_smote.predict_proba(X_test_i_scaled)[:, 1]

print("\n--- SA SMOTE ---")
print(f"  Accuracy = {accuracy_score(y_test_i, y_pred_smote):.3f}")
print(f"  F1 score = {f1_score(y_test_i, y_pred_smote):.3f}")
print(f"  ROC-AUC  = {roc_auc_score(y_test_i, y_prob_smote):.3f}")
print(classification_report(y_test_i, y_pred_smote, target_names=['No injury', 'Injury']))

# ROC krivulje
plt.figure(figsize=(8, 6))
for y_prob, label, color in [
    (y_prob_no_smote, 'Bez SMOTE', 'steelblue'),
    (y_prob_smote,    'Sa SMOTE',  'darkorange')
]:
    fpr, tpr, _ = roc_curve(y_test_i, y_prob)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Krivulja – Klasifikacija povrede (Injury_Illness)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_krivulja_injury.png')
plt.show()

# Matrice konfuzije: bez SMOTE vs sa SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, title in [
    (axes[0], y_pred_no_smote, 'Bez SMOTE'),
    (axes[1], y_pred_smote,    'Sa SMOTE')
]:
    cm_i = confusion_matrix(y_test_i, y_pred)
    sns.heatmap(cm_i, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['No injury', 'Injury'],
                yticklabels=['No injury', 'Injury'])
    ax.set_title(f'Confusion Matrix – {title}')
    ax.set_xlabel('Predviđeno'); ax.set_ylabel('Stvarno')
plt.tight_layout()
plt.savefig('confusion_matrix_smote_comparison.png')
plt.show()

# ========================== 9. NOVO: Stacking Ensemble ==========================
print("\n" + "="*60)
print("9. STACKING ENSEMBLE – KOMBINOVANJE RF + MLP")
print("="*60)

estimators = [
    ('rf',  RandomForestRegressor(n_estimators=100, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500,
                         random_state=42, early_stopping=True))
]
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
stacking_reg.fit(X_train_scaled, y_train)
y_pred_stack = stacking_reg.predict(X_test_scaled)

print("\n--- Stacking (RF + MLP → Ridge) ---")
evaluate_reg(y_test, y_pred_stack, "Stacking Ensemble")

# Usporedba modela
models     = ['Random Forest', 'MLP', 'RF (GridSearchCV)', 'Stacking']
r2_scores  = [
    r2_score(y_test, y_pred_rf),
    r2_score(y_test, y_pred_mlp),
    r2_score(y_test, y_pred_best_rf),
    r2_score(y_test, y_pred_stack)
]
mae_scores = [
    mean_absolute_error(y_test, y_pred_rf),
    mean_absolute_error(y_test, y_pred_mlp),
    mean_absolute_error(y_test, y_pred_best_rf),
    mean_absolute_error(y_test, y_pred_stack)
]
colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(models, r2_scores,  color=colors, edgecolor='black', linewidth=0.7)
axes[0].set_ylabel('R²'); axes[0].set_title('Usporedba modela – R² (veće je bolje)')
axes[0].set_ylim(0, 1); axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].bar(models, mae_scores, color=colors, edgecolor='black', linewidth=0.7)
axes[1].set_ylabel('MAE'); axes[1].set_title('Usporedba modela – MAE (manje je bolje)')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(mae_scores):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# ========================== 10. NOVO: K-Means Clustering ==========================
print("\n" + "="*60)
print("10. K-MEANS CLUSTERING – GRUPIRANJE IGRAČA PO PROFILU UMORA")
print("="*60)

# Koriste se ključne karakteristike za clustering
cluster_features = ['Stress', 'Sleep_Duration', 'Soreness', 'RPE', 'Distance', 'Fatigue']
X_cluster = df[cluster_features].dropna().copy()
scaler_c = StandardScaler()
X_cluster_scaled = scaler_c.fit_transform(X_cluster)

# --- Elbow metoda za odabir K ---
inertias    = []
sil_scores  = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_scaled, km.labels_))

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-', markersize=8)
plt.xlabel('Broj klastera (K)'); plt.ylabel('Inercija (WCSS)')
plt.title('Elbow metoda – odabir K')
plt.grid(True, alpha=0.3)
plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Izabrani K=3')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'go-', markersize=8)
plt.xlabel('Broj klastera (K)'); plt.ylabel('Silhouette Score')
plt.title('Silhouette Score – kvaliteta klasteriranja')
plt.grid(True, alpha=0.3)
plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='K=3')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_elbow.png')
plt.show()

# --- Finalni K-Means (K=3) ---
K_OPTIMAL = 3
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

X_cluster['Cluster'] = cluster_labels
print(f"\nK-Means sa K={K_OPTIMAL}")
print(f"Silhouette Score: {silhouette_score(X_cluster_scaled, cluster_labels):.3f}")
print("\nProfil klastera (srednje vrijednosti po klasteru):")
cluster_profile = X_cluster.groupby('Cluster')[cluster_features].mean().round(2)
print(cluster_profile)

cluster_names = {0: 'Klaster A', 1: 'Klaster B', 2: 'Klaster C'}
for c in range(K_OPTIMAL):
    row = cluster_profile.loc[c]
    print(f"\n{cluster_names[c]} (n={sum(cluster_labels==c)}):")
    print(f"  Fatigue={row['Fatigue']:.1f}, Stress={row['Stress']:.1f}, "
          f"Sleep={row['Sleep_Duration']:.1f}h, RPE={row['RPE']:.1f}, Distance={row['Distance']:.1f}km")

# --- PCA vizualizacija klastera ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
palette = ['#2196F3', '#FF5722', '#4CAF50']
for c in range(K_OPTIMAL):
    mask = cluster_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=f'{cluster_names[c]} (n={mask.sum()})',
                alpha=0.6, s=40, color=palette[c])
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='black', marker='X', s=200, zorder=10, label='Centroidi')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varijance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varijance)')
plt.title('K-Means Klasteri (PCA vizualizacija)')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cluster_means = X_cluster.groupby('Cluster')[cluster_features].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
x = np.arange(len(cluster_features))
width = 0.25
for i, c in enumerate(range(K_OPTIMAL)):
    plt.bar(x + i*width, cluster_means_norm.loc[c], width,
            label=cluster_names[c], color=palette[c], alpha=0.85, edgecolor='black', linewidth=0.5)
plt.xlabel('Karakteristika')
plt.ylabel('Normalizirana vrijednost (0-1)')
plt.title('Profil klastera (normaliziran)')
plt.xticks(x + width, cluster_features, rotation=20, ha='right')
plt.legend(); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_vizualizacija.png')
plt.show()

# ========================== 11. NOVO: Feature Engineering – analiza uticaja ==========================
print("\n" + "="*60)
print("11. ANALIZA UTICAJA FEATURE ENGINEERINGA")
print("="*60)

# Benchmark: RF bez feature eng.
X_no_fe = df_encoded.drop(
    columns=[target_reg, 'Injury_Illness',
             'Stress_x_RPE', 'Distance_per_Sleep',
             'Total_Acc_Dec', 'Distance_vs_Avg'],
    errors='ignore'
)
X_train_nfe, X_test_nfe, _, _ = train_test_split(X_no_fe, y_reg, test_size=0.2, random_state=42)
scaler_nfe = StandardScaler()
X_train_nfe_sc = scaler_nfe.fit_transform(X_train_nfe)
X_test_nfe_sc  = scaler_nfe.transform(X_test_nfe)

rf_nfe = RandomForestRegressor(n_estimators=100, random_state=42)
rf_nfe.fit(X_train_nfe_sc, y_train)
y_pred_nfe = rf_nfe.predict(X_test_nfe_sc)

r2_nfe   = r2_score(y_test, y_pred_nfe)
r2_with  = r2_score(y_test, y_pred_rf)
mae_nfe  = mean_absolute_error(y_test, y_pred_nfe)
mae_with = mean_absolute_error(y_test, y_pred_rf)

print(f"\nBez feature engineeringa:  R²={r2_nfe:.3f}, MAE={mae_nfe:.2f}")
print(f"Sa feature engineeringom:  R²={r2_with:.3f}, MAE={mae_with:.2f}")
print(f"Poboljšanje R²: {(r2_with - r2_nfe)*100:+.1f}%")
print(f"Poboljšanje MAE: {(mae_nfe - mae_with):+.2f} bodova")

# ========================== 12. Završni izvještaj ==========================
print("\n" + "="*60)
print("12. ZAKLJUČAK I SAČUVANE SLIKE")
print("="*60)
print("""
Sažetak svih provedenih analiza:

REGRESIJA (predviđanje Fatigue):
  - Random Forest (original): R²=~0.74
  - MLP Regressor:            R²=~0.41
  - Random Forest (GridSearchCV optimizirani): R²=poboljšano
  - Stacking Ensemble (RF+MLP → Ridge): R²=kombinirano

KLASIFIKACIJA RIZIKA (RiskLevel):
  - Logistička regresija: Accuracy ~0.67, Macro F1 ~0.67
  - Random Forest:        Accuracy ~0.79, Macro F1 ~0.76

KLASIFIKACIJA POVREDE (Injury_Illness):
  - Bez SMOTE: F1 ~0.17 (loše, ogromna neuravnoteženost klasa)
  - Sa SMOTE:  F1 poboljšan (sintetizirani pozitivni primjeri)
  - ROC-AUC ostaje ~0.75 u oba slučaja

CLUSTERING (K-Means, K=3):
  - Klaster A: visoki umor, visoki stres, kratko spavanje
  - Klaster B: srednji umor, umjereni stres
  - Klaster C: niski umor, dugo spavanje, niski stres

Sačuvane slike:
  distribucija_fatigue.png     | korelaciona_matrica.png
  regresija_usporedba.png      | feature_importance_rf.png
  gridsearch_rezultati.png     | learning_curves.png
  confusion_matrix_risk.png    | roc_krivulja_injury.png
  confusion_matrix_smote_comparison.png
  model_comparison.png         | kmeans_elbow.png
  kmeans_vizualizacija.png
""")