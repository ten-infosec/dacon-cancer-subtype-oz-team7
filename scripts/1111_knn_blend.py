"""Jaccard-kNN OOF + test, then blend search with saved LGBM."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from scipy import sparse
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)

tr=pd.read_csv('/home/claude/data_raw/train.csv'); te=pd.read_csv('/home/claude/data_raw/test.csv')
genes=ft.gene_columns(tr)
tr2=ft.fill_missing(tr,genes); te2=ft.fill_missing(te,genes)
le=LabelEncoder(); y=le.fit_transform(tr.SUBCLASS); C=len(le.classes_)
grp=ft.profile_hash(tr2,genes)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,grp))

def binmat(df):
    return sparse.csr_matrix((df[genes].astype(str).values!='WT').astype(np.float32))
Atr=binmat(tr2); Ate=binmat(te2)
na=np.asarray(Atr.sum(1)).ravel()

def knn_probs(Q, allowed_mask, k=20):
    """Q: csr queries; allowed_mask: bool over train rows usable as neighbors."""
    nq=np.asarray(Q.sum(1)).ravel()
    inter=(Q@Atr.T).toarray()
    un=nq[:,None]+na[None,:]-inter
    S=np.where(un>0, inter/np.maximum(un,1e-9), 0.0)
    S[:,~allowed_mask]=-1.0
    P=np.zeros((Q.shape[0],C))
    for i in range(Q.shape[0]):
        s=S[i]
        idx=np.argpartition(-s,k)[:k]
        w=np.maximum(s[idx],0.0)
        if w.sum()<=0: P[i]=1.0/C; continue
        np.add.at(P[i],y[idx],w)
        P[i]/=P[i].sum()
    return P

oof_k=np.zeros((len(y),C))
for kf,(trn,val) in enumerate(folds):
    allowed=np.zeros(len(y),bool); allowed[trn]=True
    oof_k[val]=knn_probs(Atr[val],allowed)
    log('knn fold',kf)
np.save('oof_knn.npy',oof_k)
log('kNN OOF macro F1:',round(f1_score(y,oof_k.argmax(1),average='macro'),4))
test_k=knn_probs(Ate,np.ones(len(y),bool))
np.save('test_knn.npy',test_k)
log('test knn saved')

A9=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy')
L=np.load('oof_lgb.npy'); K=oof_k
base=0.6*A9+0.4*T
log('base OOF:',round(f1_score(y,base.argmax(1),average='macro'),4))
best=(0,None)
for wl in (0,.05,.1,.15,.2,.25,.3):
    for wk in (0,.05,.1,.15,.2):
        if wl+wk>0.45: continue
        P=(1-wl-wk)*base + wl*L + wk*K
        s=f1_score(y,P.argmax(1),average='macro')
        if s>best[0]: best=(s,(wl,wk))
        print(f'  wl={wl:.2f} wk={wk:.2f} OOF={s:.4f}',flush=True)
log('BEST:',best)
log('DONE')
