"""Text model upgrade: word 1-2gram + char 3-5gram (x0.5) TF-IDF, LinearSVC. OOF + test probs + re-blend."""
import sys, re, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from scipy.special import softmax
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)
tr=pd.read_csv('/home/claude/data_raw/train.csv'); te=pd.read_csv('/home/claude/data_raw/test.csv')
genes=ft.gene_columns(tr); le=LabelEncoder(); y=le.fit_transform(tr['SUBCLASS']); grp=ft.profile_hash(tr,genes)
C=len(le.classes_); sub=pd.read_csv('/home/claude/data_raw/sample_submission.csv')
WT={"WT","WILDTYPE","WILD TYPE","NORMAL","NA","N/A","NAN","NONE","NULL"}
TYPE=lambda m:('FS' if 'FS' in m else 'DELINS' if 'DELINS' in m else 'DEL' if 'DEL' in m else 'INS' if 'INS' in m else 'DUP' if 'DUP' in m else 'STOP' if m.endswith('*') else 'SUB' if re.fullmatch(r'[A-Z]\d+[A-Z]',m) else 'OTHER')
def docs(frame):
    out=[]
    for row in frame[genes].itertuples(index=False,name=None):
        t=[]
        for g,v in zip(genes,row):
            if not isinstance(v,str): continue
            v=v.strip()
            if not v or v.upper() in WT: continue
            seen=set()
            for tok in re.split(r"[\s,;|]+",v.upper()):
                tok=re.sub(r'^P\.','',tok.strip())
                if not tok or tok in seen: continue
                seen.add(tok)
                m=re.sub(r"[^A-Z0-9*]+","_",tok).strip("_")
                if not m: continue
                ty=TYPE(m)
                t+= [f"gene_{g}",f"type_{ty}",f"gt_{g}_{ty}",f"mut_{m}",f"event_{g}_{m}"]
                mm=re.fullmatch(r'([A-Z])(\d+)([A-Z*])',m)
                if mm: t+=[f"aa_{mm.group(1)}>{mm.group(3)}", f"pos_{g}_{int(mm.group(2))//100}"]
        out.append(" ".join(t) if t else "no_mutation_token")
    return out
Dtr=docs(tr); Dte=docs(te); log('docs built (5-layer tokens)')
def fit_probs(tr_docs,ytr,prd_docs):
    vw=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=300000,sublinear_tf=True)
    vc=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=300000,sublinear_tf=True)
    Xa=hstack([vw.fit_transform(tr_docs),0.5*vc.fit_transform(tr_docs)]).tocsr()
    Xb=hstack([vw.transform(prd_docs),0.5*vc.transform(prd_docs)]).tocsr()
    m=LinearSVC(C=0.5,class_weight='balanced',random_state=SEED,max_iter=5000).fit(Xa,ytr)
    d=m.decision_function(Xb); al=np.zeros((Xb.shape[0],C))
    for i,c in enumerate(m.classes_): al[:,c]=d[:,i]
    return softmax(al,axis=1)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(tr,y,grp))
oof=np.zeros((len(tr),C))
for a,b in folds:
    oof[b]=fit_probs([Dtr[i] for i in a],y[a],[Dtr[i] for i in b]); log('fold done')
np.save('oof_text2.npy',oof)
log(f'text2 OOF {f1_score(y,oof.argmax(1),average="macro"):.4f} (구버전 0.3866)')
A=np.load('oof_anchor_ens3.npy')
best=(0,0)
for w in [0,.1,.2,.3,.4,.5,.6]:
    f1=f1_score(y,((1-w)*A+w*oof).argmax(1),average='macro'); log(f'  w_text2={w}: OOF {f1:.4f}')
    if f1>best[0]: best=(f1,w)
log(f'best: w={best[1]} OOF {best[0]:.4f}')
p_text=fit_probs(Dtr,y,Dte); np.save('text2_test_probs.npy',p_text)
p=np.load('ens9_test_probs.npy'); wb=best[1]
lab=le.inverse_transform(((1-wb)*p+wb*p_text).argmax(1))
s=sub.copy(); s['SUBCLASS']=lab; s.to_csv('1110_blendTxt2.csv',index=False)
log(f'-> 1110_blendTxt2.csv | STES {np.mean(lab=="STES")*100:.1f}%')
log('DONE')
