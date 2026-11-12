"""KIPAN-KIRC / GBMLGG-LGG experts trained on duplicate-free rows, OOF-validated."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from collections import Counter
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)

tr=pd.read_csv('/home/claude/data_raw/train.csv')
genes=ft.gene_columns(tr); tr2=ft.fill_missing(tr,genes)
le=LabelEncoder(); y=le.fit_transform(tr.SUBCLASS); cls=list(le.classes_)
grp=ft.profile_hash(tr2,genes)
Xtr=pd.read_pickle('Mtr.pkl')
A=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy')
oof=0.6*A+0.4*T
pred=oof.argmax(1)
log('base OOF:',round(f1_score(y,pred,average='macro'),4))

# duplicate rows (profile appears >1 time in train) = label-noise rows
vc=pd.Series(grp).value_counts(); dup_h=set(vc[vc>1].index)
isdup=np.array([h in dup_h for h in grp])
log('dup rows:',int(isdup.sum()))

folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(Xtr,y,grp))
for a_name,b_name,drop_dup in [('KIPAN','KIRC',True),('GBMLGG','LGG',True),('KIPAN','KIRC',False),('GBMLGG','LGG',False)]:
    ia,ib=cls.index(a_name),cls.index(b_name)
    top2=np.argsort(-oof,axis=1)[:,:2]
    cand=np.array([set(t)=={ia,ib} for t in top2])
    newlab=np.full(len(y),-1)
    for trn,val in folds:
        sel=trn[np.isin(y[trn],[ia,ib])]
        if drop_dup: sel=sel[~isdup[sel]]
        yy=(y[sel]==ib).astype(int)
        if len(np.unique(yy))<2: continue
        mdl=xgb.XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.07,tree_method='hist',
            random_state=SEED,n_jobs=8,eval_metric='logloss',subsample=.8,colsample_bytree=.6,reg_lambda=5.)
        mdl.fit(Xtr.iloc[sel],yy,sample_weight=compute_sample_weight('balanced',yy))
        vm=val[cand[val]]
        if len(vm)==0: continue
        p=mdl.predict_proba(Xtr.iloc[vm])[:,1]
        newlab[vm]=np.where(p>0.5,ib,ia)
    chg=newlab>=0
    pt=pred.copy(); pt[chg]=newlab[chg]
    f0=f1_score(y,pred,average='macro'); f1_=f1_score(y,pt,average='macro')
    log(f'{a_name}-{b_name} dropdup={drop_dup}: 후보 {int(cand.sum())}명, OOF {f0:.4f} -> {f1_:.4f} ({"+" if f1_>f0 else "-"}{abs(f1_-f0):.4f})')
log('DONE')
