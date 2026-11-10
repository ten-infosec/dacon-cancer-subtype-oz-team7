"""Pairwise experts for confusable pairs. For test patients whose top-2 classes form a known
confusable pair, retrain a binary model on train rows of just those 2 classes and let it decide.
Validated on OOF first; sibling-corrected rows are never touched."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'repo/dacon-cancer-subtype-oz-team7/src'); import features as ft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
SEED=42; t0=time.time()
def log(*a): print(f'[{int(time.time()-t0)}s]',*a,flush=True)
tr=pd.read_csv('/home/claude/data_raw/train.csv')
genes=ft.gene_columns(tr); le=LabelEncoder(); y=le.fit_transform(tr['SUBCLASS']); grp=ft.profile_hash(tr,genes)
C=len(le.classes_)
Mtr=pd.read_pickle('Mtr.pkl')
A=np.load('oof_anchor_ens3.npy'); T=np.load('oof_text.npy')
oof=0.6*A+0.4*T
base_pred=oof.argmax(1)
log('base OOF macro F1', round(f1_score(y,base_pred,average='macro'),4))
cls=list(le.classes_)
PAIRS=[('LUAD','LUSC'),('OV','UCEC'),('COAD','STES'),('HNSC','LUSC')]
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(Mtr,y,grp))
pred2=base_pred.copy()
for a_name,b_name in PAIRS:
    ia,ib=cls.index(a_name),cls.index(b_name)
    # candidates: top-2 = {ia,ib}
    top2=np.argsort(-oof,axis=1)[:,:2]
    cand=np.array([set(t)=={ia,ib} for t in top2])
    n=cand.sum()
    if n==0: continue
    # per-fold expert
    newlab=np.full(len(y),-1)
    for trn,val in folds:
        sel=trn[np.isin(y[trn],[ia,ib])]
        if len(sel)<40: continue
        yy=(y[sel]==ib).astype(int)
        mdl=xgb.XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.07,tree_method='hist',
            random_state=SEED,n_jobs=2,eval_metric='logloss',subsample=.8,colsample_bytree=.6,reg_lambda=5.)
        mdl.fit(Mtr.iloc[sel],yy,sample_weight=compute_sample_weight('balanced',yy))
        vmask=val[cand[val]]
        if len(vmask)==0: continue
        p=mdl.predict_proba(Mtr.iloc[vmask])[:,1]
        newlab[vmask]=np.where(p>0.5,ib,ia)
    chg=(newlab>=0)
    pred_try=pred2.copy(); pred_try[chg]=newlab[chg]
    f1_old=f1_score(y,pred2,average='macro'); f1_new=f1_score(y,pred_try,average='macro')
    log(f'{a_name}↔{b_name}: 후보 {n}명, 전담모델 적용 시 OOF {f1_old:.4f} -> {f1_new:.4f} ({"채택" if f1_new>f1_old else "기각"})')
    if f1_new>f1_old: pred2=pred_try
log('final OOF with experts', round(f1_score(y,pred2,average='macro'),4))
log('DONE')
