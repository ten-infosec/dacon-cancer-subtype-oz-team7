"""1110 submissions: (2) M300 light, (3) 9-model prob ensemble, (4) +prior correction.
M = myeongseok feature set (ai_features). Grouped 5-fold CV, SEED 42, train-only fits."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
import ai_features as af
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

SEED = 42
t0 = time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]', *a, flush=True)

tr = pd.read_csv('/home/claude/data_raw/train.csv'); te = pd.read_csv('/home/claude/data_raw/test.csv')
genes = ft.gene_columns(tr); le = LabelEncoder(); y = le.fit_transform(tr['SUBCLASS']); grp = ft.profile_hash(tr, genes)
sub = pd.read_csv('/home/claude/data_raw/sample_submission.csv')
C = len(le.classes_)

# ---- M features (cached) ----
try:
    Mtr = pd.read_pickle('Mtr.pkl'); Mte = pd.read_pickle('Mte.pkl'); log('M features loaded from cache')
except Exception:
    tr2 = ft.fill_missing(tr, genes); te2 = ft.fill_missing(te, genes)
    Mtr = af.build_mutation_features(tr2, genes); Mte = af.build_mutation_features(te2, genes)
    Mtr.to_pickle('Mtr.pkl'); Mte.to_pickle('Mte.pkl'); log('M features built', Mtr.shape)

gene_cols = [c for c in Mtr.columns if c.startswith('gene_mutated__')]
summ_cols = [c for c in Mtr.columns if not c.startswith('gene_mutated__')]
log(f'M: {len(gene_cols)} gene cols + {len(summ_cols)} summary cols')

def top_genes(frame, n=300):  # train(-fold) mutation frequency only
    return list(frame[gene_cols].sum().sort_values(ascending=False).head(n).index)

HP = {'A': dict(n_estimators=100, learning_rate=0.1, max_depth=6),
      'B': dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0)}
def model(hp, seed=SEED):
    return xgb.XGBClassifier(tree_method='hist', random_state=seed, n_jobs=2,
                             eval_metric='mlogloss', **HP[hp])

CONFIGS = [('M_hpA', None, 'A'), ('M_hpB', None, 'B'), ('M300_hpB', 300, 'B')]
def cols_for(topn, fold_frame):
    return list(Mtr.columns) if topn is None else summ_cols + top_genes(fold_frame, topn)

# ---- CV: collect OOF probs per config ----
oof = {name: np.zeros((len(tr), C)) for name, _, _ in CONFIGS}
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Mtr, y, grp))
for name, topn, hp in CONFIGS:
    for a, b in folds:
        cols = cols_for(topn, Mtr.iloc[a])
        m = model(hp); m.fit(Mtr.iloc[a][cols], y[a], sample_weight=compute_sample_weight('balanced', y[a]))
        oof[name][b] = m.predict_proba(Mtr.iloc[b][cols])
    f1 = f1_score(y, oof[name].argmax(1), average='macro')
    log(f'CV {name}: OOF macro F1 {f1:.4f}')

ens_oof = np.mean([oof[n] for n, _, _ in CONFIGS], axis=0)
log(f'CV ENS(3): OOF macro F1 {f1_score(y, ens_oof.argmax(1), average="macro"):.4f}')

# ---- prior correction (train-only): w_c = (train prior / OOF predicted prior)^alpha ----
prior = np.bincount(y, minlength=C) / len(y)
pred_prior = ens_oof.mean(0)
best = (0.0, None)
for alpha in [0, .25, .5, .75, 1.0]:
    w = (prior / np.maximum(pred_prior, 1e-9)) ** alpha
    f1 = f1_score(y, (ens_oof * w).argmax(1), average='macro')
    log(f'  alpha={alpha}: OOF {f1:.4f}')
    if f1 > best[0]: best = (f1, alpha)
alpha = best[1]; w = (prior / np.maximum(pred_prior, 1e-9)) ** alpha
log(f'prior correction: alpha={alpha} (OOF {best[0]:.4f})')

# ---- full-train fits -> test probs ----
def full_probs(topn, hp, seed):
    cols = cols_for(topn, Mtr)
    m = model(hp, seed); m.fit(Mtr[cols], y, sample_weight=compute_sample_weight('balanced', y))
    return m.predict_proba(Mte[cols])

def save(fname, labels):
    s = sub.copy(); s['SUBCLASS'] = labels; s.to_csv(fname, index=False)
    log(f'-> {fname} | STES {np.mean(labels=="STES")*100:.1f}%')

# (2) M300 light, single model seed 42
p2 = full_probs(300, 'B', SEED)
save('1110_M300_hpB.csv', le.inverse_transform(p2.argmax(1)))

# (3) ensemble: 3 configs x 3 seeds
probs = []
for name, topn, hp in CONFIGS:
    for seed in (42, 7, 2026):
        probs.append(full_probs(topn, hp, seed)); log(f'full fit {name} seed {seed} done')
p3 = np.mean(probs, axis=0)
save('1110_ENS9.csv', le.inverse_transform(p3.argmax(1)))

# (4) ensemble + prior correction
save('1110_ENS9_prior.csv', le.inverse_transform((p3 * w).argmax(1)))
np.save('ens9_test_probs.npy', p3)
log('DONE')
