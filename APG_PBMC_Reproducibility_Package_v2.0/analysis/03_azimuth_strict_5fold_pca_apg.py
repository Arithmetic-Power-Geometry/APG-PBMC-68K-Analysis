import os; os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1]; WORK=BASE/'work'; out=BASE/'results'
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score,f1_score
ADT=np.load(WORK/'ADT.npy').T.astype('float32'); cl=np.load(out/'adt_classical.npy').astype('float32'); apg=np.load(out/'adt_apg.npy').astype('float32'); rna=np.load(WORK/'refDR_emb.npy').astype('float32'); y=np.load(WORK/'celltype.l1.npy',allow_pickle=True).astype(str)
splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260807).split(np.zeros(len(y)),y)); rows=[]
def fitpred(name,Xtr,Xte,ytr,yte,fold):
 sc=StandardScaler(); a=sc.fit_transform(Xtr); b=sc.transform(Xte); m=RidgeClassifier(class_weight='balanced'); m.fit(a,ytr); pr=m.predict(b); rows.append([name,fold,balanced_accuracy_score(yte,pr),f1_score(yte,pr,average='macro')])
for fold,(tr,te) in enumerate(splits,1):
 print('fold',fold,flush=True)
 # strict fold-specific ADT PCA
 sadt=StandardScaler(); At=sadt.fit_transform(ADT[tr]); Av=sadt.transform(ADT[te]); pca=PCA(n_components=30,svd_solver='randomized',random_state=20260807+fold); Pt=pca.fit_transform(At); Pv=pca.transform(Av)
 # PCA-derived comparisons; final classifier scaling inside helper
 fitpred('ADT PCA (30)',Pt,Pv,y[tr],y[te],fold)
 fitpred('ADT PCA + classical',np.c_[Pt,cl[tr]],np.c_[Pv,cl[te]],y[tr],y[te],fold)
 fitpred('ADT PCA + APG',np.c_[Pt,apg[tr]],np.c_[Pv,apg[te]],y[tr],y[te],fold)
 fitpred('ADT PCA + classical + APG',np.c_[Pt,cl[tr],apg[tr]],np.c_[Pv,cl[te],apg[te]],y[tr],y[te],fold)
 # supplied RNA PCA is fixed reference feature; no labels used to derive it
 fitpred('RNA PCA (50)',rna[tr],rna[te],y[tr],y[te],fold)
 fitpred('RNA PCA + ADT classical',np.c_[rna[tr],cl[tr]],np.c_[rna[te],cl[te]],y[tr],y[te],fold)
 fitpred('RNA PCA + ADT APG',np.c_[rna[tr],apg[tr]],np.c_[rna[te],apg[te]],y[tr],y[te],fold)
 fitpred('RNA PCA + ADT classical + APG',np.c_[rna[tr],cl[tr],apg[tr]],np.c_[rna[te],cl[te],apg[te]],y[tr],y[te],fold)
r=pd.DataFrame(rows,columns=['model','fold','balanced_accuracy','macro_f1']);r.to_csv(out/'azimuth_strict_5fold_metrics.csv',index=False)
s=r.groupby('model').agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index(); s=s.sort_values('macro_f1_mean',ascending=False);s.to_csv(out/'azimuth_strict_5fold_summary.csv',index=False);print(s.to_string(index=False))
