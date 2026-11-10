"""Weekend scripts (태은 단독). Usage: python3 weekend.py <task> [level]
  thresh  : OOF probabilities -> per-class bias tuning (Top8-8), reports CV before/after
  light   : lightweight models: top-N genes (by train mutation frequency, fold-wise) + engineered feats
  models  : 5-model comparison on the same features / same group 5-fold
level: E7 | E8 (default E8)
All CV = StratifiedGroupKFold(5, shuffle, seed 42), groups = profile hash. Fit on train only.
"""
import sys, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

SEED = 42
task = sys.argv[1]; level = sys.argv[2] if len(sys.argv) > 2 else 'E8'
tr = pd.read_csv('/home/claude/data_raw/train.csv'); te = pd.read_csv('/home/claude/data_raw/test.csv')
genes = ft.gene_columns(tr); le = LabelEncoder(); y = le.fit_transform(tr['SUBCLASS']); K = len(le.classes_)
grp = ft.profile_hash(tr, genes)
X, Xte = ft.build_features(tr, te, level)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, grp))
def xgbA(): return xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method='hist', random_state=SEED, n_jobs=2, eval_metric='mlogloss')
def log(msg):
    print(msg, flush=True); open(f'weekend_{task}.log', 'a').write(msg + '\n')

# ------------------------------------------------------------------ thresh
if task == 'thresh':
    oof = np.zeros((len(y), K)); t0 = time.time()
    for a, b in folds:
        m = xgbA(); m.fit(X.iloc[a], y[a], sample_weight=compute_sample_weight('balanced', y[a])); oof[b] = m.predict_proba(X.iloc[b])
    np.save(f'oof_{level}.npy', oof)
    base = f1_score(y, oof.argmax(1), average='macro'); log(f'[thresh] {level} OOF Macro F1 (argmax) {base:.4f} ({int(time.time()-t0)}s)')
    # per-class additive bias on log-prob, coordinate ascent, tuned on 4 folds, evaluated on the 5th (nested, honest)
    def tune(P, yy, iters=3):
        b = np.zeros(K); L = np.log(P + 1e-9)
        for _ in range(iters):
            for k in range(K):
                best, bb = f1_score(yy, (L + b).argmax(1), average='macro'), b[k]
                for d in np.linspace(-2, 2, 21):
                    b2 = b.copy(); b2[k] = d; s = f1_score(yy, (L + b2).argmax(1), average='macro')
                    if s > best: best, bb = s, d
                b[k] = bb
        return b
    nested = []
    for a, b_ in folds:
        bias = tune(oof[a], y[a]); nested.append(f1_score(y[b_], (np.log(oof[b_] + 1e-9) + bias).argmax(1), average='macro'))
    log(f'[thresh] nested (tune on 4 folds, test on 1): {np.mean(nested):.4f} ± {np.std(nested):.4f}  vs argmax {base:.4f}')
    bias = tune(oof, y); log('[thresh] bias (full OOF): ' + json.dumps({c: round(float(v), 2) for c, v in zip(le.classes_, bias) if abs(v) > 0.05}))
    np.save(f'bias_{level}.npy', bias)
    # submission with bias applied
    m = xgbA(); m.fit(X, y, sample_weight=compute_sample_weight('balanced', y)); P = m.predict_proba(Xte)
    p = le.inverse_transform((np.log(P + 1e-9) + bias).argmax(1))
    sub = pd.read_csv('/home/claude/data_raw/sample_submission.csv'); sub['SUBCLASS'] = p; sub.to_csv(f'1107_{level}_bias.csv', index=False)
    log('[thresh] test 분포 상위: ' + str((pd.Series(p).value_counts(normalize=True) * 100).round(1).head(5).to_dict()))

# ------------------------------------------------------------------ light
elif task == 'light':
    gene_cols = [c for c in X.columns if c in set(genes)]; eng_cols = [c for c in X.columns if c not in set(genes)]
    log(f'[light] {level}: gene cols {len(gene_cols)}, engineered {len(eng_cols)}')
    for N in [0, 100, 300, 1000, len(gene_cols)]:
        sc = []; t0 = time.time()
        for a, b in folds:
            freq = X.iloc[a][gene_cols].sum(0).sort_values(ascending=False)       # fold-wise: train rows only
            keep = freq.index[:N].tolist() + eng_cols
            m = xgbA(); m.fit(X.iloc[a][keep], y[a], sample_weight=compute_sample_weight('balanced', y[a]))
            sc.append(f1_score(y[b], m.predict(X.iloc[b][keep]), average='macro'))
        log(f'[light] top-{N} genes + {len(eng_cols)} eng = {N+len(eng_cols)} cols: {np.mean(sc):.4f} ± {np.std(sc):.4f} ({int(time.time()-t0)}s)')

# ------------------------------------------------------------------ models
elif task == 'models':
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    scaler_cols = [c for c in X.columns if c not in set(genes)]      # only engineered cols need scaling
    def prep(a, b):
        Xa, Xb = X.iloc[a].copy(), X.iloc[b].copy(); sc = StandardScaler().fit(Xa[scaler_cols])
        Xa[scaler_cols] = sc.transform(Xa[scaler_cols]); Xb[scaler_cols] = sc.transform(Xb[scaler_cols]); return Xa.values, Xb.values
    models = {
        'XGBoost(A)': lambda: xgbA(),
        'LightGBM': lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, class_weight='balanced', random_state=SEED, n_jobs=2, verbose=-1),
        'L1 Logistic': lambda: LogisticRegression(penalty='l1', solver='saga', C=0.1, class_weight='balanced', max_iter=300, random_state=SEED),
        'Linear SVM': lambda: LinearSVC(C=0.05, class_weight='balanced', random_state=SEED),
        'MLP(256)': lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=60, random_state=SEED),
    }
    for name, mk in models.items():
        sc = []; t0 = time.time()
        for a, b in folds:
            Xa, Xb = prep(a, b); m = mk()
            if name.startswith('XGB'): m.fit(Xa, y[a], sample_weight=compute_sample_weight('balanced', y[a]))
            else: m.fit(Xa, y[a])
            sc.append(f1_score(y[b], m.predict(Xb), average='macro'))
        log(f'[models] {name:12s}: {np.mean(sc):.4f} ± {np.std(sc):.4f} ({int(time.time()-t0)}s)')
log('DONE ' + task)
