import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parents[1]
WORK = BASE / 'work'
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)

ADT = np.load(WORK / 'ADT.npy').T.astype('float64')
APG = np.load(OUT / 'adt_apg.npy').astype('float64')
CLASSICAL = np.load(OUT / 'adt_classical.npy').astype('float64')
y = np.load(WORK / 'celltype.l1.npy', allow_pickle=True).astype(str)

# Reviewer-requested transformation-complexity control.
# The control uses ordinary normalized ADT weights q rather than APG's
# normalized squared weights w, but applies the same 12-coordinate summary
# family: normalized entropy, max concentration, quadratic concentration,
# effective dimension, participation ratio, six exponent defects, and one
# integrated deformation. Thus APG and control are exactly dimension matched
# and closely transformation-family matched.
eps = 1e-15
S = ADT.sum(axis=1, keepdims=True)
q = np.divide(ADT, S, out=np.zeros_like(ADT), where=S > 0)
qsafe = np.where(q > 0, q, 1.0)
nfeat = ADT.shape[1]
H = -(q * np.log(qsafe)).sum(axis=1)
Hn = H / np.log(nfeat)
cmax = q.max(axis=1)
c2 = (q*q).sum(axis=1)
neff = np.exp(H)
pr = 1.0 / np.maximum(c2, eps)
ps = [2.25, 2.5, 3.0, 4.0, 5.0, 6.0]
defects = [1.0 - np.power(q, p/2.0).sum(axis=1) for p in ps]
integ = np.trapezoid(np.c_[np.zeros(len(q)), *defects], x=np.array([2.0, *ps]), axis=1)
CTRL = np.c_[Hn, cmax, c2, neff, pr, *defects, integ].astype('float64')
assert CTRL.shape[1] == APG.shape[1] == 12

np.save(OUT / 'adt_ordinary_weight_matched_control.npy', CTRL.astype('float32'))

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260807).split(np.zeros(len(y)), y))
rows = []

def fitpred(name, Xtr, Xte, ytr, yte, fold):
    sc = StandardScaler()
    a = sc.fit_transform(Xtr)
    b = sc.transform(Xte)
    model = RidgeClassifier(class_weight='balanced')
    model.fit(a, ytr)
    pred = model.predict(b)
    rows.append({
        'model': name,
        'fold': fold,
        'n_features': Xtr.shape[1],
        'balanced_accuracy': balanced_accuracy_score(yte, pred),
        'macro_f1': f1_score(yte, pred, average='macro')
    })

for fold, (tr, te) in enumerate(splits, 1):
    sadt = StandardScaler()
    Atr = sadt.fit_transform(ADT[tr])
    Ate = sadt.transform(ADT[te])
    pca = PCA(n_components=30, svd_solver='randomized', random_state=20260807 + fold)
    Ptr = pca.fit_transform(Atr)
    Pte = pca.transform(Ate)

    fitpred('ADT PCA (30)', Ptr, Pte, y[tr], y[te], fold)
    fitpred('ADT PCA + ordinary-weight matched control (42)', np.c_[Ptr, CTRL[tr]], np.c_[Pte, CTRL[te]], y[tr], y[te], fold)
    fitpred('ADT PCA + APG (42)', np.c_[Ptr, APG[tr]], np.c_[Pte, APG[te]], y[tr], y[te], fold)
    fitpred('ADT PCA + classical (37)', np.c_[Ptr, CLASSICAL[tr]], np.c_[Pte, CLASSICAL[te]], y[tr], y[te], fold)
    fitpred('ADT PCA + classical + ordinary-weight matched control (49)', np.c_[Ptr, CLASSICAL[tr], CTRL[tr]], np.c_[Pte, CLASSICAL[te], CTRL[te]], y[tr], y[te], fold)
    fitpred('ADT PCA + classical + APG (49)', np.c_[Ptr, CLASSICAL[tr], APG[tr]], np.c_[Pte, CLASSICAL[te], APG[te]], y[tr], y[te], fold)

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'azimuth_dimension_matched_metrics.csv', index=False)
summary = metrics.groupby(['model', 'n_features'], as_index=False).agg(
    balanced_accuracy_mean=('balanced_accuracy', 'mean'),
    balanced_accuracy_sd=('balanced_accuracy', 'std'),
    macro_f1_mean=('macro_f1', 'mean'),
    macro_f1_sd=('macro_f1', 'std'))
summary.to_csv(OUT / 'azimuth_dimension_matched_summary.csv', index=False)

comparisons = [
    ('APG42_vs_control42', 'ADT PCA + APG (42)', 'ADT PCA + ordinary-weight matched control (42)'),
    ('APG49_vs_control49', 'ADT PCA + classical + APG (49)', 'ADT PCA + classical + ordinary-weight matched control (49)')
]
paired = []
for label, a_name, b_name in comparisons:
    a = metrics[metrics.model == a_name].sort_values('fold').macro_f1.to_numpy()
    b = metrics[metrics.model == b_name].sort_values('fold').macro_f1.to_numpy()
    d = a - b
    mean = d.mean()
    sem = stats.sem(d)
    ci = stats.t.interval(0.95, len(d)-1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
    w = stats.wilcoxon(d, alternative='two-sided', zero_method='wilcox')
    paired.append({
        'comparison': label,
        'mean_macro_f1_difference_apg_minus_control': mean,
        'ci95_low_fold_level_t': ci[0],
        'ci95_high_fold_level_t': ci[1],
        'wilcoxon_statistic': w.statistic,
        'wilcoxon_p_two_sided': w.pvalue,
        'all_folds_apg_higher': bool(np.all(d > 0)),
        'fold_differences': ';'.join(f'{x:.8f}' for x in d)
    })
pd.DataFrame(paired).to_csv(OUT / 'azimuth_dimension_matched_paired.csv', index=False)

print(summary.to_string(index=False))
print(pd.DataFrame(paired).to_string(index=False))
