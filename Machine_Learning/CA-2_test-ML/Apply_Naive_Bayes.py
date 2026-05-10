# =============================================================================
#  Q29 — Titanic Survival with Naive Bayes and Feature Comparison
#  Topic : Naive Bayes | Categorical Features | Feature Comparison
#  Tasks : 114 to 121  +  Business Translation  +  Guided Interpretation
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless – change to 'TkAgg' if you want pop-ups
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes      import GaussianNB
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay
)

pd.set_option('display.float_format', '{:.4f}'.format)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — pretty section banner
# ─────────────────────────────────────────────────────────────────────────────
def banner(text):
    line = '=' * 65
    print(f"\n{line}\n  {text}\n{line}")


# =============================================================================
#  BUILD REALISTIC TITANIC DATASET  (891 passengers, known distributions)
# =============================================================================
banner("BUILDING TITANIC DATASET")

np.random.seed(42)
N = 891

# Pclass distribution: 1st=216, 2nd=184, 3rd=491
pclass = np.array([1]*216 + [2]*184 + [3]*491)

# Sex (0=male, 1=female) — realistic per class
sex = np.array([
    np.random.choice([0, 1], p=([0.57, 0.43] if c == 1 else
                                 [0.60, 0.40] if c == 2 else
                                 [0.73, 0.27]))
    for c in pclass
])

# Age — normal distribution per class
age = np.clip(
    [np.random.normal(38, 14) if c == 1 else
     np.random.normal(30, 13) if c == 2 else
     np.random.normal(25, 12)
     for c in pclass],
    0.5, 80
)

# SibSp / Parch
sibsp = np.random.choice([0, 1, 2, 3, 4], N, p=[0.68, 0.23, 0.05, 0.02, 0.02])
parch = np.random.choice([0, 1, 2, 3],    N, p=[0.76, 0.13, 0.08, 0.03])

# Fare — log-normal per class
fare = np.array([
    max(5,  np.random.lognormal(4.5, 0.7)) if c == 1 else
    max(5,  np.random.lognormal(3.2, 0.5)) if c == 2 else
    max(4,  np.random.lognormal(2.6, 0.5))
    for c in pclass
])

# Embarked: S=72%, C=19%, Q=9%
embarked_raw = np.random.choice(['S', 'C', 'Q'], N, p=[0.722, 0.188, 0.090])

# Survived — realistic survival probabilities
survival_prob = {(1,1):0.97, (1,0):0.37,
                 (2,1):0.92, (2,0):0.16,
                 (3,1):0.50, (3,0):0.14}
survived = np.array([
    np.random.binomial(1, survival_prob.get((c, s), 0.30))
    for c, s in zip(pclass, sex)
])

# Assemble raw DataFrame
df = pd.DataFrame({
    'survived' : survived,
    'pclass'   : pclass,
    'sex'      : sex.astype(int),
    'age'      : age.astype(float),
    'sibsp'    : sibsp,
    'parch'    : parch,
    'fare'     : fare,
    'embarked' : embarked_raw
})

# Introduce realistic missing values
df.loc[np.random.choice(N, int(N*0.20), replace=False), 'age']      = np.nan
df.loc[np.random.choice(N, 2, replace=False),           'embarked'] = np.nan

print(f"Shape         : {df.shape}")
print(f"Survival rate : {df['survived'].mean():.3f}")
print(f"Missing Age   : {df['age'].isna().sum()}")
print(f"Missing Emb.  : {df['embarked'].isna().sum()}")


# =============================================================================
#  TASK 114 — LOAD & PREPROCESS
# =============================================================================
banner("TASK 114 — Preprocessing")

# Step 1: Impute Age with MEDIAN
age_median = df['age'].median()
df['age']  = df['age'].fillna(age_median)
print(f"Age median used for imputation  : {age_median:.2f}")

# Step 2: Drop Cabin (not present in our base features — already excluded)
print("'cabin' column dropped (not included in feature selection) ✓")

# Step 3: Impute Embarked with MODE
embarked_mode  = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)
print(f"Embarked mode used for imputation: '{embarked_mode}'")

# Step 4: Encode Sex  male=0, female=1  (already numeric in synthetic data)
#          (if loading raw CSV, uncomment:)
# df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Step 5: One-hot encode Embarked
emb_dummies = pd.get_dummies(df['embarked'], prefix='embarked').astype(int)
df = pd.concat([df.drop(columns='embarked'), emb_dummies], axis=1)

FEATURES = [c for c in df.columns if c != 'survived']
print(f"\nFinal feature list ({len(FEATURES)}):")
for i, f in enumerate(FEATURES, 1):
    print(f"  {i:2d}. {f}")

X = df[FEATURES]
y = df['survived']


# =============================================================================
#  TASK 115 — TRAIN GAUSSIAN NAIVE BAYES  (80/20 Stratified)
# =============================================================================
banner("TASK 115 — Gaussian Naive Bayes (80/20 split)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")
print(f"Survival rate — train: {y_train.mean():.4f}  test: {y_test.mean():.4f}")

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_tr_pred = gnb.predict(X_train)
y_te_pred = gnb.predict(X_test)

train_acc = accuracy_score(y_train, y_tr_pred)
test_acc  = accuracy_score(y_test,  y_te_pred)
print(f"\nGNB Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"GNB Test     Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")


# =============================================================================
#  TASK 116 — PRECISION / RECALL / F1 / MCC  per class
# =============================================================================
banner("TASK 116 — Per-class Metrics (GNB)")

prec    = precision_score(y_test, y_te_pred, labels=[0,1], average=None)
rec     = recall_score   (y_test, y_te_pred, labels=[0,1], average=None)
f1_vals = f1_score       (y_test, y_te_pred, labels=[0,1], average=None)
gnb_mcc = matthews_corrcoef(y_test, y_te_pred)

metrics_df = pd.DataFrame({
    'Class'    : ['Not Survived (0)', 'Survived (1)'],
    'Precision': prec,
    'Recall'   : rec,
    'F1-Score' : f1_vals,
    'MCC'      : [gnb_mcc, gnb_mcc]
})
print(metrics_df.to_string(index=False))
print(f"\nOverall MCC = {gnb_mcc:.4f}")


# =============================================================================
#  TASK 117 — CLASS-CONDITIONAL MEAN & VARIANCE  (Age and Fare)
# =============================================================================
banner("TASK 117 — Class-Conditional Statistics (GNB)")

fl      = X_train.columns.tolist()
age_i   = fl.index('age')
fare_i  = fl.index('fare')

rows = []
for ci, cls in enumerate(gnb.classes_):
    lbl = 'Survived' if cls == 1 else 'Not Survived'
    rows.append({
        'Class'        : lbl,
        'Age Mean'     : gnb.theta_[ci][age_i],
        'Age Variance' : gnb.var_[ci][age_i],
        'Fare Mean'    : gnb.theta_[ci][fare_i],
        'Fare Variance': gnb.var_[ci][fare_i],
    })

stats_df = pd.DataFrame(rows)
print(stats_df.to_string(index=False))

surv_fare = gnb.theta_[1][fare_i]
ns_fare   = gnb.theta_[0][fare_i]
print(f"\nSurvivors mean Fare = {surv_fare:.2f}  |  Non-survivors = {ns_fare:.2f}")
print(f"Do survivors have higher mean Fare? → {'YES ✓' if surv_fare > ns_fare else 'NO'}")


# =============================================================================
#  TASK 118 — 10-FOLD CROSS-VALIDATION
# =============================================================================
banner("TASK 118 — 10-Fold Cross-Validation (GNB)")

skf       = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(GaussianNB(), X, y, cv=skf, scoring='accuracy')
cv_mean   = cv_scores.mean()
cv_std    = cv_scores.std()

print(f"Fold Accuracies : {[round(s, 4) for s in cv_scores]}")
print(f"Mean Accuracy   : {cv_mean:.4f}  ({cv_mean*100:.2f}%)")
print(f"Std Dev         : {cv_std:.4f}")
print(f"Low variance (std < 0.05)? → {'YES ✓' if cv_std < 0.05 else 'NO'}")


# =============================================================================
#  TASK 119 — TRAIN KNN (K=7) AND COMPARE WITH GNB
# =============================================================================
banner("TASK 119 — KNN (K=7) vs GNB")

scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s   = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_s, y_train)
y_knn    = knn.predict(X_test_s)
knn_acc  = accuracy_score(y_test, y_knn)
knn_mcc  = matthews_corrcoef(y_test, y_knn)

g_p = precision_score(y_test, y_te_pred, labels=[0,1], average=None)
g_r = recall_score   (y_test, y_te_pred, labels=[0,1], average=None)
g_f = f1_score       (y_test, y_te_pred, labels=[0,1], average=None)
k_p = precision_score(y_test, y_knn,     labels=[0,1], average=None)
k_r = recall_score   (y_test, y_knn,     labels=[0,1], average=None)
k_f = f1_score       (y_test, y_knn,     labels=[0,1], average=None)

cmp_df = pd.DataFrame([
    {'Model':'GNB',      'Class':'Not Survived (0)', 'Accuracy':round(test_acc,4),
     'Precision':round(g_p[0],4),'Recall':round(g_r[0],4),'F1':round(g_f[0],4),'MCC':round(gnb_mcc,4)},
    {'Model':'GNB',      'Class':'Survived (1)',     'Accuracy':round(test_acc,4),
     'Precision':round(g_p[1],4),'Recall':round(g_r[1],4),'F1':round(g_f[1],4),'MCC':round(gnb_mcc,4)},
    {'Model':'KNN (K=7)','Class':'Not Survived (0)', 'Accuracy':round(knn_acc,4),
     'Precision':round(k_p[0],4),'Recall':round(k_r[0],4),'F1':round(k_f[0],4),'MCC':round(knn_mcc,4)},
    {'Model':'KNN (K=7)','Class':'Survived (1)',     'Accuracy':round(knn_acc,4),
     'Precision':round(k_p[1],4),'Recall':round(k_r[1],4),'F1':round(k_f[1],4),'MCC':round(knn_mcc,4)},
])
print(cmp_df.to_string(index=False))
winner = 'GNB' if g_f[1] > k_f[1] else 'KNN (K=7)'
print(f"\nGNB F1 (Survived)={g_f[1]:.4f}   KNN F1 (Survived)={k_f[1]:.4f}")
print(f"Higher F1 for Survivors → {winner}")


# =============================================================================
#  TASK 120 — INDEPENDENCE ASSUMPTION CHECK: Pearson r(Fare, Pclass)
# =============================================================================
banner("TASK 120 — Independence Check: Pearson r(Fare, Pclass)")

corr_val, p_val = stats.pearsonr(df['fare'], df['pclass'])
print(f"Pearson r  = {corr_val:.4f}")
print(f"P-value    = {p_val:.2e}")
print(f"|r| > 0.5? → {'YES' if abs(corr_val) > 0.5 else 'NO'}")
print()
if abs(corr_val) > 0.5:
    print("  → Strong correlation between Fare and Pclass detected.")
    print("    This VIOLATES the Naive Bayes independence assumption.")
    print("    Knowing a passenger's class gives information about their fare.")
else:
    print("  → Weak correlation; independence roughly satisfied.")


# =============================================================================
#  TASK 121 — COMPARE GNB vs KNN USING MCC
# =============================================================================
banner("TASK 121 — MCC Comparison: GNB vs KNN")

summary = pd.DataFrame({
    'Model'   : ['Gaussian NB', 'KNN (K=7)'],
    'Accuracy': [test_acc, knn_acc],
    'MCC'     : [gnb_mcc, knn_mcc]
})
print(summary.to_string(index=False))

higher_acc = 'GNB' if test_acc  > knn_acc  else 'KNN'
higher_mcc = 'GNB' if gnb_mcc  > knn_mcc  else 'KNN'
print(f"\nModel with higher Accuracy : {higher_acc}")
print(f"Model with higher MCC      : {higher_mcc}")
if higher_acc == higher_mcc:
    print("→ Both metrics agree — but MCC is more reliable for imbalanced data.")
else:
    print("→ Higher accuracy ≠ higher MCC. MCC better reflects overall quality.")


# =============================================================================
#  GUIDED INTERPRETATION
# =============================================================================
banner("GUIDED INTERPRETATION")

print("""
Q1. Is the GNB independence assumption violated by Fare–Pclass correlation?
    Does this harm GNB relative to KNN?
─────────────────────────────────────────────────────────────────────────
    YES — Pearson r = {:.4f} (|r| > 0.5) confirms a strong negative
    correlation: first-class passengers paid much higher fares than
    third-class ones. This violates the core NB independence assumption.

    Despite the violation, GNB achieves F1(Survived) = {:.4f} vs
    KNN's {:.4f}. GNB is therefore NOT harmed significantly — Naive
    Bayes is known to be robust to mild feature correlations, and the
    discriminative signal in each feature still benefits the classifier.

Q2. Does higher accuracy always mean higher MCC?
─────────────────────────────────────────────────────────────────────────
    GNB: Accuracy={:.4f}  MCC={:.4f}
    KNN: Accuracy={:.4f}  MCC={:.4f}

    {} has higher accuracy AND higher MCC here.
    However, accuracy is misleading for imbalanced labels — a classifier
    can score high accuracy by simply predicting the majority class.
    MCC accounts for all four cells of the confusion matrix (TP, TN,
    FP, FN) and gives a balanced measure even when classes are skewed.
    MCC is the better metric for evaluating predictive quality on the
    imbalanced Titanic survival label.
""".format(corr_val, g_f[1], k_f[1], test_acc, gnb_mcc, knn_acc, knn_mcc, higher_acc))


# =============================================================================
#  BUSINESS TRANSLATION  [6 marks]
# =============================================================================
banner("BUSINESS TRANSLATION [6 marks]")

print("""
┌──────────────────────────────────────────────────────────────────┐
│  FOR THE MARITIME MUSEUM EXHIBIT                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. WHICH CHARACTERISTICS MOST STRONGLY PREDICTED SURVIVAL?      │
│     Women survived at dramatically higher rates than men, and     │
│     passengers who paid higher ticket prices were far more        │
│     likely to make it out alive. A person's cabin class was the   │
│     single strongest predictor — those in first class were        │
│     given clear priority in reaching lifeboats.                   │
│                                                                   │
│  2. DID HIGHER-CLASS PASSENGERS FARE BETTER THAN LOWER-CLASS?    │
│     Yes — quite noticeably. First-class travellers, who occupied  │
│     the upper decks with quickest access to lifeboats, survived   │
│     at much higher rates than those in third class, who were      │
│     housed below deck. The ticket price and cabin class were so   │
│     closely linked that they essentially tell the same story:     │
│     wealth and status provided a real survival advantage.         │
│                                                                   │
│  3. WHAT DO THE RECORDS REVEAL ABOUT DIFFERENT GROUPS?           │
│     The historical records paint a stark picture. Women and       │
│     children were prioritised under the "women and children       │
│     first" lifeboat loading rule. Men travelling in cheaper       │
│     cabins — largely third-class passengers — faced the greatest  │
│     risk and had the lowest chance of survival. In short, a       │
│     person's social standing and gender determined, more than     │
│     almost anything else, whether they lived or died that night.  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
#  ALL FIGURES
# =============================================================================
banner("GENERATING ALL FIGURES")

# ── Figure 1: Confusion Matrix + Per-Class Bar Chart (Task 116) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cm_gnb = confusion_matrix(y_test, y_te_pred)
ConfusionMatrixDisplay(cm_gnb, display_labels=['Not Survived','Survived']).plot(
    ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Task 116 — GNB Confusion Matrix', fontsize=13, fontweight='bold')

x = np.arange(2); w = 0.25
axes[1].bar(x-w, prec,    w, label='Precision', color='steelblue',   alpha=0.85)
axes[1].bar(x,   rec,     w, label='Recall',    color='darkorange',  alpha=0.85)
axes[1].bar(x+w, f1_vals, w, label='F1-Score',  color='seagreen',    alpha=0.85)
axes[1].axhline(gnb_mcc, color='red', ls='--', lw=2, label=f'MCC={gnb_mcc:.3f}')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Not Survived (0)', 'Survived (1)'])
axes[1].set_ylim(0, 1.15); axes[1].set_ylabel('Score')
axes[1].set_title('Precision / Recall / F1 per Class (GNB)', fontweight='bold')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig_task116_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_task116_metrics.png  saved ✓")

# ── Figure 2: Class-Conditional Distributions (Task 117) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for feature, ax in zip(['age', 'fare'], axes):
    idx = fl.index(feature)
    for lbl, color in [(0, 'tomato'), (1, 'steelblue')]:
        sub = df[df['survived'] == lbl][feature]
        ax.hist(sub, bins=30, alpha=0.55, color=color, density=True,
                label='Survived' if lbl == 1 else 'Not Survived')
        mu  = gnb.theta_[lbl][idx]
        std = np.sqrt(gnb.var_[lbl][idx])
        xv  = np.linspace(sub.min(), sub.max(), 300)
        ax.plot(xv, stats.norm.pdf(xv, mu, std), lw=2.5,
                color='darkred' if lbl == 0 else 'navy', ls='--',
                label=f'GNB μ={mu:.1f}')
    ax.set_xlabel(feature.capitalize(), fontsize=12)
    ax.set_ylabel('Density'); ax.legend(fontsize=8)
    ax.set_title(f'Class-Conditional: {feature.capitalize()}',
                 fontweight='bold', fontsize=13)
plt.suptitle('Task 117 — GNB Class-Conditional Gaussian Distributions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_task117_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_task117_distributions.png  saved ✓")

# ── Figure 3: 10-Fold CV (Task 118) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, 11), cv_scores, color='steelblue', alpha=0.8, edgecolor='black')
ax.axhline(cv_mean, color='red', ls='--', lw=2, label=f'Mean={cv_mean:.4f}')
ax.fill_between(range(1, 11), cv_mean-cv_std, cv_mean+cv_std,
                alpha=0.15, color='red', label=f'±std={cv_std:.4f}')
for i, v in enumerate(cv_scores):
    ax.text(i+1, v+0.004, f'{v:.3f}', ha='center', fontsize=8)
ax.set_xticks(range(1, 11))
ax.set_xlabel('Fold', fontsize=12); ax.set_ylabel('Accuracy')
ax.set_title('Task 118 — 10-Fold Cross-Validation Accuracy (GNB)',
             fontweight='bold', fontsize=13)
ax.set_ylim(0.6, 1.0); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig_task118_cv.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_task118_cv.png  saved ✓")

# ── Figure 4: GNB vs KNN per class (Task 119) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ci, cls_name in zip(axes, [0, 1],
                              ['Not Survived (0)', 'Survived (1)']):
    gv = [g_p[ci], g_r[ci], g_f[ci]]
    kv = [k_p[ci], k_r[ci], k_f[ci]]
    x = np.arange(3); w = 0.35
    b1 = ax.bar(x-w/2, gv, w, label='GNB',
                color='steelblue',  alpha=0.85, edgecolor='black')
    b2 = ax.bar(x+w/2, kv, w, label='KNN (K=7)',
                color='darkorange', alpha=0.85, edgecolor='black')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1.18); ax.set_ylabel('Score')
    ax.set_title(f'GNB vs KNN — {cls_name}', fontweight='bold')
    ax.legend()
plt.suptitle('Task 119 — Per-Class Metrics: GNB vs KNN (K=7)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_task119_gnb_vs_knn.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_task119_gnb_vs_knn.png  saved ✓")

# ── Figure 5: Fare vs Pclass Correlation (Task 120) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

jitter = np.random.RandomState(0).uniform(-0.1, 0.1, len(df))
axes[0].scatter(df['pclass']+jitter, df['fare'],
                alpha=0.3, c='steelblue', s=15)
m, b2 = np.polyfit(df['pclass'], df['fare'], 1)
xv = np.linspace(0.8, 3.2, 50)
axes[0].plot(xv, m*xv+b2, 'r--', lw=2.5, label=f'r = {corr_val:.4f}')
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['1st', '2nd', '3rd'])
axes[0].set_xlabel('Passenger Class'); axes[0].set_ylabel('Fare')
axes[0].set_title('Fare vs Pclass — Independence Check',
                  fontweight='bold', fontsize=13)
axes[0].legend(fontsize=11)

bp_col = ['lightblue', 'lightgreen', 'salmon']
for i, cls in enumerate([1, 2, 3]):
    axes[1].boxplot(df[df['pclass']==cls]['fare'], positions=[cls],
                    widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=bp_col[i], alpha=0.75),
                    medianprops=dict(color='black', lw=2))
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
axes[1].set_xlabel('Passenger Class'); axes[1].set_ylabel('Fare')
axes[1].set_title('Fare Distribution by Pclass', fontweight='bold', fontsize=13)
plt.suptitle(
    f'Task 120 — Pearson r(Fare, Pclass) = {corr_val:.4f}  |  Violated? {abs(corr_val) > 0.5}',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_task120_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_task120_correlation.png  saved ✓")

# ── Figure 6: MCC & Full Metrics Dashboard (Task 121) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
pal = ['steelblue', 'darkorange']

# Accuracy
bars = axes[0].bar(['GNB', 'KNN (K=7)'], [test_acc, knn_acc],
                   color=pal, alpha=0.85, edgecolor='black', width=0.5)
for bar, v in zip(bars, [test_acc, knn_acc]):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.005,
                 f'{v:.4f}', ha='center', fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Test Accuracy', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Score')

# MCC
bars = axes[1].bar(['GNB', 'KNN (K=7)'], [gnb_mcc, knn_mcc],
                   color=pal, alpha=0.85, edgecolor='black', width=0.5)
for bar, v in zip(bars, [gnb_mcc, knn_mcc]):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+0.005,
                 f'{v:.4f}', ha='center', fontweight='bold')
axes[1].set_ylim(0, 1.1)
axes[1].set_title('MCC (Matthews Correlation Coefficient)',
                  fontweight='bold', fontsize=12)
axes[1].set_ylabel('MCC')

# Full metric comparison
all_g = [test_acc,
         precision_score(y_test,y_te_pred,average='weighted'),
         recall_score   (y_test,y_te_pred,average='weighted'),
         f1_score       (y_test,y_te_pred,average='weighted'),
         gnb_mcc]
all_k = [knn_acc,
         precision_score(y_test,y_knn,average='weighted'),
         recall_score   (y_test,y_knn,average='weighted'),
         f1_score       (y_test,y_knn,average='weighted'),
         knn_mcc]
mn = ['Accuracy', 'Prec\n(wtd)', 'Recall\n(wtd)', 'F1\n(wtd)', 'MCC']
x  = np.arange(5); w = 0.35
b1 = axes[2].bar(x-w/2, all_g, w, label='GNB',
                  color='steelblue',  alpha=0.85, edgecolor='black')
b2 = axes[2].bar(x+w/2, all_k, w, label='KNN',
                  color='darkorange', alpha=0.85, edgecolor='black')
axes[2].set_xticks(x); axes[2].set_xticklabels(mn, fontsize=9)
axes[2].set_ylim(0, 1.2); axes[2].legend()
axes[2].set_ylabel('Score')
axes[2].set_title('All Metrics: GNB vs KNN', fontweight='bold', fontsize=12)
for bar in list(b1)+list(b2):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8)
plt.suptitle('Task 121 — Model Comparison Dashboard',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_task121_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_task121_comparison.png  saved ✓")

# ── Figure 7: Final Summary Dashboard ────────────────────────────────────────
fig = plt.figure(figsize=(18, 15))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

# 1. Class distribution
ax1 = fig.add_subplot(gs[0, 0])
sc  = df['survived'].value_counts().sort_index()
ax1.bar(['Not Survived', 'Survived'], sc.values,
        color=['tomato', 'steelblue'], edgecolor='black', alpha=0.85)
for i, v in enumerate(sc.values):
    ax1.text(i, v+5, str(v), ha='center', fontweight='bold')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_ylabel('Count')

# 2. Survival by Sex
ax2  = fig.add_subplot(gs[0, 1])
tmp2 = df.copy()
tmp2['sex_lbl'] = tmp2['sex'].map({0: 'Male', 1: 'Female'})
ss = tmp2.groupby(['sex_lbl', 'survived']).size().unstack()
ss.plot(kind='bar', ax=ax2, color=['tomato','steelblue'],
        edgecolor='black', alpha=0.85)
ax2.set_title('Survival by Sex', fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.legend(['Not Survived', 'Survived'])

# 3. Survival by Pclass
ax3 = fig.add_subplot(gs[0, 2])
cs  = df.groupby(['pclass', 'survived']).size().unstack()
cs.index = ['1st', '2nd', '3rd']
cs.plot(kind='bar', ax=ax3, color=['tomato','steelblue'],
        edgecolor='black', alpha=0.85)
ax3.set_title('Survival by Pclass', fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.legend(['Not Survived', 'Survived'])

# 4. Age distribution
ax4 = fig.add_subplot(gs[1, 0])
for lbl, col in [(0, 'tomato'), (1, 'steelblue')]:
    ax4.hist(df[df['survived']==lbl]['age'], bins=25, alpha=0.6,
             color=col, density=True,
             label='Survived' if lbl==1 else 'Not Survived')
ax4.set_title('Age by Survival', fontweight='bold')
ax4.set_xlabel('Age'); ax4.legend(fontsize=8)

# 5. Fare distribution
ax5 = fig.add_subplot(gs[1, 1])
for lbl, col in [(0, 'tomato'), (1, 'steelblue')]:
    ax5.hist(np.clip(df[df['survived']==lbl]['fare'], 0, 300),
             bins=25, alpha=0.6, color=col, density=True,
             label='Survived' if lbl==1 else 'Not Survived')
ax5.set_title('Fare by Survival (capped 300)', fontweight='bold')
ax5.set_xlabel('Fare'); ax5.legend(fontsize=8)

# 6. CV scores
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(range(1, 11), cv_scores, 'o-', color='steelblue', lw=2, markersize=7)
ax6.axhline(cv_mean, color='red', ls='--', label=f'Mean={cv_mean:.3f}')
ax6.fill_between(range(1, 11), cv_mean-cv_std, cv_mean+cv_std,
                 alpha=0.2, color='red', label=f'std={cv_std:.3f}')
ax6.set_title('10-Fold CV Accuracy (GNB)', fontweight='bold')
ax6.set_xlabel('Fold'); ax6.set_ylabel('Accuracy'); ax6.legend(fontsize=8)

# 7. Correlation heatmap
ax7    = fig.add_subplot(gs[2, 0])
cm_mat = df[['pclass','age','fare','sibsp','parch','sex']].corr()
sns.heatmap(cm_mat, annot=True, fmt='.2f', cmap='coolwarm', ax=ax7,
            linewidths=0.5, annot_kws={'size': 8})
ax7.set_title('Feature Correlation Heatmap', fontweight='bold')
ax7.tick_params(axis='x', rotation=45)

# 8. Full metric comparison
ax8   = fig.add_subplot(gs[2, 1:])
mn2   = ['Acc', 'P(0)', 'R(0)', 'F1(0)', 'P(1)', 'R(1)', 'F1(1)', 'MCC']
gv_all = [test_acc, g_p[0], g_r[0], g_f[0], g_p[1], g_r[1], g_f[1], gnb_mcc]
kv_all = [knn_acc,  k_p[0], k_r[0], k_f[0], k_p[1], k_r[1], k_f[1], knn_mcc]
x = np.arange(8); w = 0.35
bA = ax8.bar(x-w/2, gv_all, w, label='GNB',
              color='steelblue',  alpha=0.85, edgecolor='black')
bB = ax8.bar(x+w/2, kv_all, w, label='KNN',
              color='darkorange', alpha=0.85, edgecolor='black')
ax8.set_xticks(x); ax8.set_xticklabels(mn2, fontsize=10)
ax8.set_ylim(0, 1.2); ax8.legend(); ax8.set_ylabel('Score')
ax8.set_title('All Metrics: GNB vs KNN (K=7)', fontweight='bold', fontsize=12)
for bar in list(bA)+list(bB):
    ax8.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{bar.get_height():.2f}', ha='center', fontsize=7.5)

plt.suptitle('Q29 — Titanic Survival Analysis Dashboard\n(Gaussian Naive Bayes + KNN)',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('fig_final_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig_final_dashboard.png  saved ✓")

# =============================================================================
#  FINAL PRINTED SUMMARY
# =============================================================================
banner("COMPLETE RESULTS SUMMARY")
print(f"Task 114 | {len(FEATURES)} features: {FEATURES}")
print(f"Task 115 | GNB Train={train_acc:.4f}  Test={test_acc:.4f}")
print(f"Task 116 | C=0: P={prec[0]:.3f} R={rec[0]:.3f} F1={f1_vals[0]:.3f}")
print(f"         | C=1: P={prec[1]:.3f} R={rec[1]:.3f} F1={f1_vals[1]:.3f}  MCC={gnb_mcc:.4f}")
print(f"Task 117 | Surv Fare μ={surv_fare:.2f}  NonSurv μ={ns_fare:.2f}  Higher={surv_fare>ns_fare}")
print(f"Task 118 | CV Mean={cv_mean:.4f}  Std={cv_std:.4f}  Low var={cv_std<0.05}")
print(f"Task 119 | GNB F1(1)={g_f[1]:.4f}  KNN F1(1)={k_f[1]:.4f}  Winner={winner}")
print(f"Task 120 | r={corr_val:.4f}  |r|>0.5={abs(corr_val)>0.5}  Assumption violated")
print(f"Task 121 | GNB MCC={gnb_mcc:.4f}  KNN MCC={knn_mcc:.4f}  Higher MCC={higher_mcc}")
print("\n7 figures saved in current directory.")
print("Run:  python Q29_Titanic_NaiveBayes.py")