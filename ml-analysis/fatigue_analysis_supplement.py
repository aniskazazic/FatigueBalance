import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Workout_Routine_Dirty.csv')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

RISK_COLORS = {'low': '#4CAF50', 'medium': '#FFC107', 'high': '#FF5722', 'critical': '#B71C1C'}
RISK_ORDER = ['low', 'medium', 'high', 'critical']
total = len(df)
print(f"Dataset: {total} rows")

# ========== A. Frekvencije ==========
risk_counts = df['RiskLevel'].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
print("\nRiskLevel frequencies:")
print(risk_counts)

# Plot risk level
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(RISK_ORDER, risk_counts.values, color=[RISK_COLORS[r] for r in RISK_ORDER], edgecolor='black')
for bar, val in zip(axes[0].patches, risk_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')
axes[0].set_ylabel('Count'); axes[0].set_title('RiskLevel counts'); axes[0].grid(axis='y', alpha=0.3)
axes[1].pie(risk_counts.values, labels=RISK_ORDER, colors=[RISK_COLORS[r] for r in RISK_ORDER], autopct='%1.1f%%', startangle=90)
plt.tight_layout(); plt.savefig('A1_risklevel_frekvencija.png'); plt.close()

# Category frequencies
sess_counts = df['Session_Type'].value_counts()
pos_counts = df['Position'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(sess_counts.index, sess_counts.values, color='#5C6BC0', edgecolor='black')
for i, (_, v) in enumerate(sess_counts.items()):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
axes[0].set_title('Session_Type'); axes[0].grid(axis='y', alpha=0.3)
axes[1].barh(pos_counts.index[::-1], pos_counts.values[::-1], color='#26A69A', edgecolor='black')
axes[1].set_title('Position (top 10)')
plt.tight_layout(); plt.savefig('A2A3_kategoricke_frekvencije.png'); plt.close()

# Injury
inj_counts = df['Injury_Illness'].value_counts()
print("\nInjury_Illness:")
print(inj_counts)

# ========== B. Deskriptivna statistika + skewness/kurtosis ==========
desc = df[num_cols].describe().T
desc['skewness'] = df[num_cols].skew()
desc['kurtosis'] = df[num_cols].kurt()
desc['median'] = df[num_cols].median()
desc['IQR'] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
print("\nExtended descriptive stats (skewness, kurtosis):")
print(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']].round(3))

# Skewness bar plot
plt.figure(figsize=(12, 5))
colors_skew = ['#E53935' if abs(s) > 1 else '#FB8C00' if abs(s) > 0.5 else '#43A047' for s in desc['skewness']]
bars = plt.barh(desc.index, desc['skewness'], color=colors_skew, edgecolor='black')
plt.axvline(0, color='black', linewidth=1.5)
plt.axvline(0.5, color='orange', linestyle='--', alpha=0.7); plt.axvline(-0.5, color='orange', linestyle='--', alpha=0.7)
plt.axvline(1.0, color='red', linestyle='--', alpha=0.7); plt.axvline(-1.0, color='red', linestyle='--', alpha=0.7)
for bar, s in zip(bars, desc['skewness']):
    plt.text(s + (0.02 if s >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
             f'{s:.3f}', va='center', ha='left' if s >= 0 else 'right', fontsize=8.5)
plt.xlabel('Skewness'); plt.title('Skewness of numeric variables'); plt.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig('B_skewness_sve_varijable.png'); plt.close()

# ========== C. Sleep duration analysis ==========
SLEEP_THRESHOLD = 9.0
df['Sleep_Preporuka'] = df['Sleep_Duration'].apply(lambda x: 'NEDOVOLJNO (<9h)' if x < SLEEP_THRESHOLD else 'DOVOLJNO (≥9h)')
sleep_counts = df['Sleep_Preporuka'].value_counts()
print("\nSleep recommendation:")
print(sleep_counts)

# Bins
bins = [0, 5, 6, 7, 8, 9, 15]
labels = ['<5h', '5-6h', '6-7h', '7-8h', '8-9h', '>9h']
df['Sleep_Bin'] = pd.cut(df['Sleep_Duration'], bins=bins, labels=labels, right=False)
bin_counts = df['Sleep_Bin'].value_counts().reindex(labels)
print("\nSleep bin counts:")
print(bin_counts)

# Cross-tab
ct = pd.crosstab(df['Sleep_Bin'], df['RiskLevel'], margins=True, margins_name='UKUPNO')
print("\nSleep bin × RiskLevel:")
print(ct)

# T-test
g1 = df[df['Sleep_Preporuka'] == 'NEDOVOLJNO (<9h)']['Fatigue'].dropna()
g2 = df[df['Sleep_Preporuka'] == 'DOVOLJNO (≥9h)']['Fatigue'].dropna()
t_stat, p_val = stats.ttest_ind(g1, g2)
print(f"\nT-test (insufficient vs sufficient sleep): t={t_stat:.3f}, p={p_val:.6f}")

# Sleep visualizations
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['Sleep_Duration'], bins=30, color='#5C6BC0', alpha=0.85, edgecolor='white')
ax1.axvline(SLEEP_THRESHOLD, color='red', lw=2.5, ls='--', label=f'Threshold {SLEEP_THRESHOLD}h')
ax1.axvline(df['Sleep_Duration'].mean(), color='orange', lw=2, label=f'Mean {df["Sleep_Duration"].mean():.2f}h')
ax1.set_xlabel('Sleep Duration (h)'); ax1.set_ylabel('Count'); ax1.set_title('Sleep duration distribution'); ax1.legend(); ax1.grid(alpha=0.3)
ax2 = fig.add_subplot(gs[0, 2])
ax2.pie(sleep_counts.values, labels=sleep_counts.index, colors=['#E53935', '#43A047'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Sleep recommendation')
ax3 = fig.add_subplot(gs[1, 0])
colors_bins = ['#B71C1C', '#E53935', '#FF7043', '#43A047', '#1E88E5', '#1565C0']
ax3.bar(labels, bin_counts.values, color=colors_bins, edgecolor='black')
for i, v in enumerate(bin_counts.values):
    ax3.text(i, v + 1, str(v), ha='center', fontweight='bold')
ax3.set_xlabel('Sleep category'); ax3.set_ylabel('Count'); ax3.set_title('Sleep bin distribution'); ax3.grid(axis='y', alpha=0.3)
ax4 = fig.add_subplot(gs[1, 1])
bp = ax4.boxplot([g1.values, g2.values], labels=sleep_counts.index, patch_artist=True, medianprops={'color':'black','lw':2})
bp['boxes'][0].set_facecolor('#E53935'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#43A047'); bp['boxes'][1].set_alpha(0.7)
ax4.set_ylabel('Fatigue'); ax4.set_title('Fatigue by sleep recommendation'); ax4.grid(axis='y', alpha=0.3)
ax5 = fig.add_subplot(gs[1, 2])
crosstab_plot = pd.crosstab(df['Sleep_Bin'], df['RiskLevel']).reindex(labels).reindex(columns=RISK_ORDER, fill_value=0)
bottom = np.zeros(len(labels))
for risk_cat in RISK_ORDER:
    vals = crosstab_plot[risk_cat].values.astype(float)
    ax5.bar(labels, vals, bottom=bottom, color=RISK_COLORS[risk_cat], label=risk_cat, edgecolor='white', lw=0.5)
    bottom += vals
ax5.set_xlabel('Sleep category'); ax5.set_ylabel('Count'); ax5.set_title('Sleep × RiskLevel (stacked)'); ax5.legend(); ax5.grid(axis='y', alpha=0.3)
plt.savefig('C_sleep_analiza.png', dpi=150, bbox_inches='tight'); plt.close()

# ========== D. Kombinacije kategorija ==========
print("\nCategory combinations:")
combos = {
    'LOW vs ostali': ('low', ['medium','high','critical']),
    'LOW+MEDIUM vs HIGH+CRITICAL': (['low','medium'], ['high','critical']),
    'LOW+MEDIUM+HIGH vs CRITICAL': (['low','medium','high'], ['critical'])
}
for name, (grp_a, grp_b) in combos.items():
    a = df[df['RiskLevel'].isin(grp_a)] if isinstance(grp_a, list) else df[df['RiskLevel'] == grp_a]
    b = df[df['RiskLevel'].isin(grp_b)] if isinstance(grp_b, list) else df[df['RiskLevel'] == grp_b]
    print(f"{name:30} | A: {len(a):4d} | B: {len(b):4d} | Fatigue A: {a['Fatigue'].mean():.2f} | B: {b['Fatigue'].mean():.2f} | p={stats.ttest_ind(a['Fatigue'].dropna(), b['Fatigue'].dropna())[1]:.4f}")

# Additional combo: LOW+MEDIUM, HIGH, CRITICAL
g_lm = df[df['RiskLevel'].isin(['low','medium'])]['Fatigue'].dropna()
g_h = df[df['RiskLevel'] == 'high']['Fatigue'].dropna()
g_c = df[df['RiskLevel'] == 'critical']['Fatigue'].dropna()
n_lm, n_h, n_c = len(g_lm), len(g_h), len(g_c)
print(f"\nLOW+MEDIUM: {n_lm} | HIGH: {n_h} | CRITICAL: {n_c}")
print(f"T-test LM vs H: p={stats.ttest_ind(g_lm, g_h)[1]:.4f}")
print(f"T-test LM vs C: p={stats.ttest_ind(g_lm, g_c)[1]:.4f}")
print(f"T-test H vs C : p={stats.ttest_ind(g_h, g_c)[1]:.4f}")

# Visualize D2c
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
groups_3 = ['LOW+MEDIUM', 'HIGH', 'CRITICAL']
values_3 = [n_lm, n_h, n_c]
colors_3 = ['#43A047', '#FF5722', '#B71C1C']
means_3 = [g_lm.mean(), g_h.mean(), g_c.mean()]
for ax, data, title, ylabel in zip(axes, [values_3, means_3, [g_lm, g_h, g_c]],
                                   ['Count', 'Mean Fatigue', 'Fatigue distribution'],
                                   ['Count', 'Mean Fatigue', 'Fatigue']):
    if title == 'Count':
        ax.bar(groups_3, data, color=colors_3, edgecolor='black')
        for i, v in enumerate(data):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
        ax.set_ylim(0, max(values_3)*1.2)
    elif title == 'Mean Fatigue':
        ax.bar(groups_3, data, color=colors_3, edgecolor='black')
        for i, v in enumerate(data):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
        ax.set_ylim(0, max(means_3)*1.2)
    else:
        bp = ax.boxplot(data, patch_artist=True, medianprops={'color':'black','lw':2})
        for patch, color in zip(bp['boxes'], colors_3):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticklabels(groups_3)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('D2c_lowmed_high_critical.png'); plt.close()

# ========== E. Distribucije svih numeričkih varijabli ==========
n_vars = len(num_cols)
fig, axes = plt.subplots(n_vars, 3, figsize=(16, n_vars * 3.2))
if n_vars == 1:
    axes = [axes]  # ensure 2D indexing works
for i, col in enumerate(num_cols):
    data = df[col].dropna()
    skw = data.skew()
    color = '#E53935' if abs(skw) > 1 else '#FB8C00' if abs(skw) > 0.5 else '#43A047'
    # histogram
    ax_h = axes[i, 0]
    ax_h.hist(data, bins=25, color=color, alpha=0.65, edgecolor='white', density=True)
    kde_x = np.linspace(data.min(), data.max(), 200)
    ax_h.plot(kde_x, stats.gaussian_kde(data)(kde_x), 'k-', lw=2)
    ax_h.axvline(data.mean(), color='blue', lw=1.5, ls='--')
    ax_h.axvline(data.median(), color='purple', lw=1.5, ls=':')
    ax_h.set_title(f'{col}\nskew={skw:.3f} | kurt={data.kurt():.3f}')
    ax_h.grid(alpha=0.3)
    # boxplot
    ax_b = axes[i, 1]
    bp = ax_b.boxplot(data, vert=True, patch_artist=True, medianprops={'color':'black','lw':2})
    bp['boxes'][0].set_facecolor(color); bp['boxes'][0].set_alpha(0.7)
    ax_b.set_title(f'Boxplot: {col}')
    ax_b.set_xticklabels([col])
    ax_b.grid(axis='y', alpha=0.3)
    # Q-Q
    ax_q = axes[i, 2]
    qq = stats.probplot(data, dist='norm')
    ax_q.scatter(qq[0][0], qq[0][1], alpha=0.4, s=12, color=color)
    slope, intercept, r = qq[1][0], qq[1][1], qq[1][2]
    x_line = np.array([qq[0][0][0], qq[0][0][-1]])
    ax_q.plot(x_line, slope * x_line + intercept, 'r-', lw=2, label=f'r={r:.3f}')
    ax_q.set_xlabel('Theoretical quantiles'); ax_q.set_ylabel('Observed quantiles')
    ax_q.set_title(f'Q-Q: {col}'); ax_q.legend(); ax_q.grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('E_distribucija_sve_varijable.png', dpi=130)
plt.close()

# ========== F. Sažetak asimetrije ==========
skew_df = pd.DataFrame({
    'Variable': num_cols,
    'Skewness': [df[c].skew() for c in num_cols],
    'Kurtosis': [df[c].kurt() for c in num_cols]
}).sort_values('Skewness')
print("\nSkewness summary (sorted):")
print(skew_df.to_string(index=False))

# Grid of histograms
fig, axes = plt.subplots(3, 5, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    data = df[col].dropna()
    skw = data.skew()
    c = '#E53935' if abs(skw) > 1 else '#FB8C00' if abs(skw) > 0.5 else '#43A047'
    axes[i].hist(data, bins=22, color=c, alpha=0.7, edgecolor='white', density=True)
    kde_x = np.linspace(data.min(), data.max(), 200)
    axes[i].plot(kde_x, stats.gaussian_kde(data)(kde_x), 'k-', lw=1.8)
    axes[i].set_title(f'{col}\nskew={skw:.3f}', fontsize=8.5)
    axes[i].grid(alpha=0.3)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout(); plt.savefig('F_grid_asimetrija.png'); plt.close()

# Boxplots normalized
fig, ax = plt.subplots(figsize=(16, 6))
data_norm = []
for col in num_cols:
    d = df[col].dropna()
    data_norm.append(((d - d.mean()) / d.std()).values)
bp = ax.boxplot(data_norm, labels=num_cols, patch_artist=True, medianprops={'color':'black','lw':1.5})
colors_bp = ['#E53935' if abs(df[c].skew()) > 1 else '#FB8C00' if abs(df[c].skew()) > 0.5 else '#43A047' for c in num_cols]
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xticklabels(num_cols, rotation=30, ha='right')
ax.set_ylabel('Z-score'); ax.set_title('Boxplots (Z-score normalized)'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('F2_boxplots_usporedba.png'); plt.close()

print("\nAll plots saved.")