"""Small, honest tuning of the two new views (token-kNN k, NB alpha). Fold-safe; judged by half-split honest gain."""
import sys, re, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)
exec(open('1111_views.py').read().split("def vote")[0].split("SEED=42; t0=time.time()")[1].split("def log")[0])
tr=pd.read_csv('/home/claude/data_raw/train.csv'); te=pd.read_csv('/home/claude/data_raw/test.csv')
genes=ft.gene_columns(tr); tr2=ft.fill_missing(tr,genes)
y=LabelEncoder().fit_transform(tr.SUBCLASS); C=26
grp=ft.profile_hash(tr2,genes)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,grp))
src=open('1111_views.py').read()
exec(src[src.index('WT={'):src.index('Dtr=np.array')])
Dtr=np.array(docs(tr),dtype=object); Dte=np.array(docs(te),dtype=object); log('docs')
def vote(S,ytr,k):
    P=np.zeros((S.shape[0],C))
    for i in range(S.shape[0]):
        s=S[i]; idx=np.argpartition(-s,k)[:k]; w=np.maximum(s[idx],0)
        if w.sum()<=0: P[i]=1/C; continue
        np.add.at(P[i],ytr[idx],w); P[i]/=P[i].sum()
    return P
# token-kNN: compute similarity once per fold, vote with several k
KS=[10,20,40]; oofk={k:np.zeros((len(y),C)) for k in KS}; tstk={}
for trn,val in folds:
    vec=TfidfVectorizer(min_df=1,sublinear_tf=True); A=vec.fit_transform(Dtr[trn]); S=(vec.transform(Dtr[val])@A.T).toarray()
    for k in KS: oofk[k][val]=vote(S,y[trn],k)
vec=TfidfVectorizer(min_df=1,sublinear_tf=True); A=vec.fit_transform(Dtr); S=(vec.transform(Dte)@A.T).toarray()
for k in KS: tstk[k]=vote(S,y,k); np.save(f'oof_v_kt{k}.npy',oofk[k]); np.save(f'test_v_kt{k}.npy',tstk[k])
log('token-kNN done')
AL=[0.1,0.5,1.0]; oofn={a:np.zeros((len(y),C)) for a in AL}
for trn,val in folds:
    vec=TfidfVectorizer(min_df=2,sublinear_tf=True,use_idf=False,norm=None); A=vec.fit_transform(Dtr[trn]); Q=vec.transform(Dtr[val])
    for a in AL:
        m=ComplementNB(alpha=a).fit(A,y[trn]); P=np.zeros((Q.shape[0],C)); P[:,m.classes_]=m.predict_proba(Q); oofn[a][val]=P
vec=TfidfVectorizer(min_df=2,sublinear_tf=True,use_idf=False,norm=None); A=vec.fit_transform(Dtr); Q=vec.transform(Dte)
for a in AL:
    m=ComplementNB(alpha=a).fit(A,y); P=np.zeros((Q.shape[0],C)); P[:,m.classes_]=m.predict_proba(Q)
    np.save(f'oof_v_nb{a}.npy',oofn[a]); np.save(f'test_v_nb{a}.npy',P)
log('NB done')
A9=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy'); K=np.load('oof_knn.npy')
cur=0.85*(0.6*A9+0.4*T)+0.15*K
f=lambda P,idx=slice(None): f1_score(y[idx],P[idx].argmax(1),average='macro')
def blend(k,a,wt=0.15,wn=0.05): return (1-wt-wn)*cur+wt*oofk[k]+wn*oofn[a]
base=f(blend(20,0.5)); log(f'후보 A (k20, a0.5) OOF {base:.4f}')
rng=np.random.RandomState(42); splits=[]
for r in range(3):
    idx=rng.permutation(len(y)); h=len(y)//2; splits+= [(idx[:h],idx[h:]),(idx[h:],idx[:h])]
for k in KS:
    for a in AL:
        P=blend(k,a); s=f(P)
        g=[f(P,b)-f(blend(20,0.5),b) for _,b in splits]
        print(f'  k={k:2d} alpha={a:<4} OOF {s:.4f} ({s-base:+.4f}) | 절반평가 대비A {np.mean(g):+.4f} (양수 {sum(x>0 for x in g)}/6)',flush=True)
log('DONE')
