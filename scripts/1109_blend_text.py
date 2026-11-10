"""Anchor(ENS, LB 0.354) + text TF-IDF LinearSVC diversity blend. Weight chosen on OOF only."""
import sys, re, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
import ai_features as af
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.special import softmax
import xgboost as xgb

SEED = 42; t0 = time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]', *a, flush=True)

tr = pd.read_csv('/home/claude/data_raw/train.csv'); te = pd.read_csv('/home/claude/data_raw/test.csv')
genes = ft.gene_columns(tr); le = LabelEncoder(); y = le.fit_transform(tr['SUBCLASS']); grp = ft.profile_hash(tr, genes)
sub = pd.read_csv('/home/claude/data_raw/sample_submission.csv'); C = len(le.classes_)
Mtr = pd.read_pickle('Mtr.pkl'); Mte = pd.read_pickle('Mte.pkl')
gene_cols = [c for c in Mtr.columns if c.startswith('gene_mutated__')]
summ_cols = [c for c in Mtr.columns if not c.startswith('gene_mutated__')]
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Mtr, y, grp))

# ---- text documents (Myeongseok 04 rule: gene / mut / event tokens) ----
WT = {"WT","WILDTYPE","WILD TYPE","NORMAL","NA","N/A","NAN","NONE","NULL"}
def docs(frame):
    out = []
    for row in frame[genes].itertuples(index=False, name=None):
        terms = []
        for gene, v in zip(genes, row):
            if not isinstance(v, str): continue
            v = v.strip()
            if not v or v.upper() in WT: continue
            seen = set()
            for tok in re.split(r"[\s,;|]+", v.upper()):
                tok = tok.strip()
                if not tok or tok in seen: continue
                seen.add(tok)
                m = re.sub(r"[^A-Z0-9*]+", "_", tok).strip("_")
                if m: terms.extend((f"gene_{gene}", f"mut_{m}", f"event_{gene}_{m}"))
        out.append(" ".join(terms) if terms else "no_mutation_token")
    return out
Dtr = docs(tr); Dte = docs(te); log('documents built')

def text_probs(train_docs, ytr, pred_docs):
    vec = TfidfVectorizer(min_df=2, max_features=250000, sublinear_tf=True)
    Xa = vec.fit_transform(train_docs); Xb = vec.transform(pred_docs)
    m = LinearSVC(C=0.5, class_weight='balanced', random_state=SEED, max_iter=5000).fit(Xa, ytr)
    d = m.decision_function(Xb)
    al = np.zeros((Xb.shape[0], C));
    for i, c in enumerate(m.classes_): al[:, c] = d[:, i]
    return softmax(al, axis=1)

# ---- OOF: anchor ENS(3 configs) + text ----
HP = {'A': dict(n_estimators=100, learning_rate=0.1, max_depth=6),
      'B': dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0)}
def xgbm(hp): return xgb.XGBClassifier(tree_method='hist', random_state=SEED, n_jobs=2, eval_metric='mlogloss', **HP[hp])
CONFIGS = [(None,'A'), (None,'B'), (300,'B')]
oof_anchor = np.zeros((len(tr), C)); oof_text = np.zeros((len(tr), C))
for a, b in folds:
    ps = []
    for topn, hp in CONFIGS:
        cols = list(Mtr.columns) if topn is None else summ_cols + list(Mtr.iloc[a][gene_cols].sum().sort_values(ascending=False).head(topn).index)
        m = xgbm(hp); m.fit(Mtr.iloc[a][cols], y[a], sample_weight=compute_sample_weight('balanced', y[a]))
        ps.append(m.predict_proba(Mtr.iloc[b][cols]))
    oof_anchor[b] = np.mean(ps, axis=0)
    oof_text[b] = text_probs([Dtr[i] for i in a], y[a], [Dtr[i] for i in b])
    log('fold done')
np.save('oof_anchor_ens3.npy', oof_anchor); np.save('oof_text.npy', oof_text)
log(f'OOF anchor {f1_score(y, oof_anchor.argmax(1), average="macro"):.4f} | text {f1_score(y, oof_text.argmax(1), average="macro"):.4f}')

best = (0, 0)
for w in [0, .05, .1, .15, .2, .25, .3, .4]:
    f1 = f1_score(y, ((1-w)*oof_anchor + w*oof_text).argmax(1), average='macro')
    log(f'  w_text={w}: OOF {f1:.4f}')
    if f1 > best[0]: best = (f1, w)
f1b, wb = best; log(f'best blend: w_text={wb} OOF {f1b:.4f}')

# ---- test: anchor probs already saved (9-model ENS), text full fit ----
p_anchor = np.load('ens9_test_probs.npy')
p_text = text_probs(Dtr, y, Dte); np.save('text_test_probs.npy', p_text)
lab = le.inverse_transform(((1-wb)*p_anchor + wb*p_text).argmax(1))
s = sub.copy(); s['SUBCLASS'] = lab; s.to_csv('1110_blendTxt.csv', index=False)
log(f'-> 1110_blendTxt.csv | STES {np.mean(lab=="STES")*100:.1f}% | differs from ENS9: {(lab != le.inverse_transform(p_anchor.argmax(1))).sum()}')
log('DONE')
