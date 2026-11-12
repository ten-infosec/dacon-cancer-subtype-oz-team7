"""Diversity models: LightGBM + Jaccard-kNN, OOF with the same fold scheme (group=profile hash)."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy import sparse
import lightgbm as lgb
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)

tr=pd.read_csv('/home/claude/data_raw/train.csv'); te=pd.read_csv('/home/claude/data_raw/test.csv')
genes=ft.gene_columns(tr)
tr2=ft.fill_missing(tr,genes); te2=ft.fill_missing(te,genes)
le=LabelEncoder(); y=le.fit_transform(tr.SUBCLASS); C=len(le.classes_)
grp=ft.profile_hash(tr2,genes)
Xtr=pd.read_pickle('Mtr.pkl'); Xte=pd.read_pickle('Mte.pkl')
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(Xtr,y,grp))

# ---- LightGBM OOF + test ----
oof_l=np.zeros((len(y),C)); test_l=np.zeros((len(te),C))
for k,(trn,val) in enumerate(folds):
    m=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=63,min_child_samples=10,
        colsample_bytree=0.5,subsample=0.8,subsample_freq=1,reg_lambda=5.0,
        objective='multiclass',num_class=C,random_state=SEED,n_jobs=4,verbose=-1)
    m.fit(Xtr.iloc[trn],y[trn],sample_weight=compute_sample_weight('balanced',y[trn]))
    oof_l[val]=m.predict_proba(Xtr.iloc[val]); test_l+=m.predict_proba(Xte)/5
    log('lgb fold',k,'f1',round(f1_score(y[val],oof_l[val].argmax(1),average='macro'),4))
np.save('oof_lgb.npy',oof_l); np.save('test_lgb.npy',test_l)
log('LGB OOF macro F1:',round(f1_score(y,oof_l.argmax(1),average='macro'),4))

# ---- Jaccard kNN OOF + test ----
gc=genes
def binmat(df):
    V=(df[gc].astype(str).values!='WT').astype(np.float32)
    return sparse.csr_matrix(V)
Atr=binmat(tr2); Ate=binmat(te2)
na=np.asarray(Atr.sum(1)).ravel()
def knn_probs(Q,nq,excl_rows=None,k=20):
    inter=(Q@Atr.T).toarray()
    un=nq[:,None]+na[None,:]-inter
    S=np.where(un>0,inter/np.maximum(un,1e-9),0.0)
    if excl_rows is not None: S[np.arange(S.shape[0])[:,None]*0+0, :] = S  # noop
    P=np.zeros((Q.shape[0],C))
    for i in range(Q.shape[0]):
        s=S[i].copy()
        if excl_rows is not None: s[excl_rows[i]]=-1
        idx=np.argpartition(-s,k)[:k]
        w=np.maximum(s[idx],0)
        if w.sum()==0: P[i]=1.0/C; continue
        for j,ww in zip(idx,w): P[i,y[j]]+=ww
        P[i]/=P[i].sum()
    return P
# OOF: for val rows exclude all rows of val fold (and same-group rows are in same fold by construction)
oof_k=np.zeros((len(y),C))
nq=np.asarray(Atr.sum(1)).ravel()
for k_,(trn,val) in enumerate(folds):
    mask=np.ones(len(y),bool); mask[trn]=False  # exclude val-fold rows as neighbors
    excl=[np.where(mask)[0]]*len(val)
    P=knn_probs(Atr[val],nq[val],excl_rows=excl)
    oof_k[val]=P; log('knn fold',k_)
np.save('oof_knn.npy',oof_k)
log('kNN OOF macro F1:',round(f1_score(y,oof_k.argmax(1),average='macro'),4))
test_k=knn_probs(Ate,np.asarray(Ate.sum(1)).ravel())
np.save('test_knn.npy',test_k)

# ---- blend search (text pinned near 0.4 of non-tree mass; explore adding new views) ----
A9=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy')
base=0.6*A9+0.4*T
log('base OOF:',round(f1_score(y,base.argmax(1),average='macro'),4))
best=(0,None)
for wl in (0,.05,.1,.15,.2,.3):
    for wk in (0,.05,.1,.15,.2):
        if wl+wk>=0.5: continue
        rest=1-wl-wk
        P=rest*(0.6*A9+0.4*T)+wl*oof_l+wk*oof_k
        s=f1_score(y,P.argmax(1),average='macro')
        if s>best[0]: best=(s,(wl,wk))
        print(f'  wl={wl} wk={wk} OOF={s:.4f}')
log('best:',best)
