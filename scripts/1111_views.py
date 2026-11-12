"""New 'views' for the blend, all fold-safe (vocab/IDF fit on train-fold only). OOF + test probs."""
import sys, re, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from scipy import sparse
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)
tr=pd.read_csv('/home/claude/data_raw/train.csv'); te=pd.read_csv('/home/claude/data_raw/test.csv')
genes=ft.gene_columns(tr); tr2=ft.fill_missing(tr,genes); te2=ft.fill_missing(te,genes)
le=LabelEncoder(); y=le.fit_transform(tr.SUBCLASS); C=len(le.classes_)
grp=ft.profile_hash(tr2,genes)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,grp))
WT={"WT","WILDTYPE","WILD TYPE","NORMAL","NA","N/A","NAN","NONE","NULL"}
def docs(frame):
    out=[]
    for row in frame[genes].itertuples(index=False,name=None):
        terms=[]
        for gene,v in zip(genes,row):
            if not isinstance(v,str): continue
            v=v.strip()
            if not v or v.upper() in WT: continue
            seen=set()
            for tok in re.split(r"[\s,;|]+",v.upper()):
                tok=tok.strip()
                if not tok or tok in seen: continue
                seen.add(tok); m=re.sub(r"[^A-Z0-9*]+","_",tok).strip("_")
                if m: terms.extend((f"gene_{gene}",f"mut_{m}",f"event_{gene}_{m}"))
        out.append(" ".join(terms) if terms else "no_mutation_token")
    return out
Dtr=np.array(docs(tr),dtype=object); Dte=np.array(docs(te),dtype=object); log('docs')
Btr=sparse.csr_matrix((tr2[genes].astype(str).values!='WT').astype(np.float32))
Bte=sparse.csr_matrix((te2[genes].astype(str).values!='WT').astype(np.float32))

def vote(S,ytr,k):
    P=np.zeros((S.shape[0],C))
    for i in range(S.shape[0]):
        s=S[i]; idx=np.argpartition(-s,k)[:k]; w=np.maximum(s[idx],0)
        if w.sum()<=0: P[i]=1/C; continue
        np.add.at(P[i],ytr[idx],w); P[i]/=P[i].sum()
    return P

def run(name, fn):
    oof=np.zeros((len(y),C))
    for trn,val in folds: oof[val]=fn(trn,val,None)
    tst=fn(np.arange(len(y)),None,True)
    np.save(f'oof_v_{name}.npy',oof); np.save(f'test_v_{name}.npy',tst)
    log(f'{name}: OOF macroF1 {f1_score(y,oof.argmax(1),average="macro"):.4f}')

# view 1: IDF-weighted cosine kNN on gene presence (rare genes count more), k=20
def knn_idf(trn,val,test,k=20):
    A=Btr[trn]; df=np.asarray(A.sum(0)).ravel(); idf=np.log((1+len(trn))/(1+df))+1
    D=sparse.diags(idf.astype(np.float32))
    Q=(Bte if test else Btr[val])@D; Aw=A@D
    S=(normalize(Q)@normalize(Aw).T).toarray()
    return vote(S,y[trn],k)
# view 2: cosine kNN on TF-IDF of mutation documents (exact variant tokens), k=20
def knn_tfidf(trn,val,test,k=20):
    vec=TfidfVectorizer(min_df=1,sublinear_tf=True); A=vec.fit_transform(Dtr[trn])
    Q=vec.transform(Dte if test else Dtr[val])
    S=(Q@A.T).toarray(); return vote(S,y[trn],k)
# view 3: Complement Naive Bayes on token counts (generative, very different inductive bias)
def cnb(trn,val,test):
    vec=TfidfVectorizer(min_df=2,sublinear_tf=True,use_idf=False,norm=None); A=vec.fit_transform(Dtr[trn])
    Q=vec.transform(Dte if test else Dtr[val])
    m=ComplementNB(alpha=0.5).fit(A,y[trn]); P=np.zeros((Q.shape[0],C)); P[:,m.classes_]=m.predict_proba(Q); return P
# view 4: logistic regression on TF-IDF (probabilistic linear, vs LinearSVC hinge)
def lr(trn,val,test):
    vec=TfidfVectorizer(min_df=2,max_features=250000,sublinear_tf=True); A=vec.fit_transform(Dtr[trn])
    Q=vec.transform(Dte if test else Dtr[val])
    m=LogisticRegression(C=8.0,class_weight='balanced',max_iter=300,solver='liblinear').fit(A,y[trn])
    P=np.zeros((Q.shape[0],C)); P[:,m.classes_]=m.predict_proba(Q); return P

for name,fn in [('knnidf',knn_idf),('knntfidf',knn_tfidf),('cnb',cnb),('lr',lr)]:
    run(name,fn)

A9=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy'); K=np.load('oof_knn.npy')
cur=0.85*(0.6*A9+0.4*T)+0.15*K
b=f1_score(y,cur.argmax(1),average='macro'); log(f'CURRENT (LB 0.4533) OOF {b:.4f}')
for name in ['knnidf','knntfidf','cnb','lr']:
    V=np.load(f'oof_v_{name}.npy'); line=[]
    for w in (0.05,0.10,0.15,0.20):
        s=f1_score(y,((1-w)*cur+w*V).argmax(1),average='macro'); line.append(f'w{w:.2f}={s:.4f}({s-b:+.4f})')
    print(f'  +{name}: '+'  '.join(line),flush=True)
log('DONE')
