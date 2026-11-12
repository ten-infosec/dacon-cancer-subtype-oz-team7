"""Mahalanobis-distance classifiers (shrinkage LDA) as an extra blend view. Fold-safe."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from scipy import sparse
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)
tr=pd.read_csv('/home/claude/data_raw/train.csv'); te=pd.read_csv('/home/claude/data_raw/test.csv')
genes=ft.gene_columns(tr); tr2=ft.fill_missing(tr,genes); te2=ft.fill_missing(te,genes)
y=LabelEncoder().fit_transform(tr.SUBCLASS); C=26
grp=ft.profile_hash(tr2,genes)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,grp))
Xtr=pd.read_pickle('Mtr.pkl'); Xte=pd.read_pickle('Mte.pkl')
# summary (non-gene) features only: patient-level columns are the strong signal
gene_set=set(genes)
summ=[c for c in Xtr.columns if c not in gene_set]
log('summary cols',len(summ),'all cols',Xtr.shape[1])
Btr=sparse.csr_matrix((tr2[genes].astype(str).values!='WT').astype(np.float32))
Bte=sparse.csr_matrix((te2[genes].astype(str).values!='WT').astype(np.float32))

def run(name, fn):
    oof=np.zeros((len(y),C))
    for trn,val in folds: oof[val]=fn(trn,val,False)
    tst=fn(np.arange(len(y)),None,True)
    np.save(f'oof_v_{name}.npy',oof); np.save(f'test_v_{name}.npy',tst)
    log(f'{name}: OOF macroF1 {f1_score(y,oof.argmax(1),average="macro"):.4f}')

def lda_summary(trn,val,test):
    A=Xtr[summ].values[trn].astype(np.float64); Q=(Xte[summ].values if test else Xtr[summ].values[val]).astype(np.float64)
    sc=StandardScaler().fit(A); A=np.nan_to_num(sc.transform(A)); Q=np.nan_to_num(sc.transform(Q))
    m=LDA(solver='lsqr',shrinkage='auto',priors=np.full(C,1/C)).fit(A,y[trn])
    return m.predict_proba(Q)
def lda_svd(trn,val,test):
    svd=TruncatedSVD(100,random_state=SEED).fit(Btr[trn])
    A=svd.transform(Btr[trn]); Q=svd.transform(Bte if test else Btr[val])
    A2=np.hstack([A,np.log1p(np.asarray(Btr[trn].sum(1)))]); Q2=np.hstack([Q,np.log1p(np.asarray((Bte if test else Btr[val]).sum(1)))])
    sc=StandardScaler().fit(A2)
    m=LDA(solver='lsqr',shrinkage='auto',priors=np.full(C,1/C)).fit(sc.transform(A2),y[trn])
    return m.predict_proba(sc.transform(Q2))
for n,f in [('lda_summ',lda_summary),('lda_svd',lda_svd)]: run(n,f)

A9=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy'); K=np.load('oof_knn.npy')
KT=np.load('oof_v_knntfidf.npy'); NB=np.load('oof_v_cnb.npy')
cur=0.85*(0.6*A9+0.4*T)+0.15*K
cand=0.80*cur+0.15*KT+0.05*NB
f=lambda P:f1_score(y,P.argmax(1),average='macro')
for base_name,base in [('현재 0.4533 조합',cur),('내일 후보 A',cand)]:
    b=f(base); log(f'{base_name} OOF {b:.4f}')
    for name in ['lda_summ','lda_svd']:
        V=np.load(f'oof_v_{name}.npy')
        print(f'   +{name}: '+'  '.join(f'{w:.2f}:{f((1-w)*base+w*V)-b:+.4f}' for w in (0.05,0.10,0.15,0.20)),flush=True)
log('DONE')
