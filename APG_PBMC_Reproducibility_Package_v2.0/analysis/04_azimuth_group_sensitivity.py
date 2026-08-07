import os; os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1]; WORK=BASE/'work'; out=BASE/'results'
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score
from scipy.stats import wilcoxon
ADT=np.load(WORK/'ADT.npy').T.astype('float32'); cl=np.load(out/'adt_classical.npy').astype('float32'); apg=np.load(out/'adt_apg.npy').astype('float32'); rna=np.load(WORK/'refDR_emb.npy').astype('float32'); y=np.load(WORK/'celltype.l1.npy',allow_pickle=True).astype(str); cn=np.load(WORK/'cellnames.npy',allow_pickle=True).astype(str); groups=np.array([x.split('_',1)[0] for x in cn])
rows=[]
def pred(name,A,B,tr,te):
 sc=StandardScaler(); At=sc.fit_transform(A); Bt=sc.transform(B); m=RidgeClassifier(class_weight='balanced');m.fit(At,y[tr]); pr=m.predict(Bt);rows.append([groups[te[0]],len(te),name,balanced_accuracy_score(y[te],pr),f1_score(y[te],pr,average='macro')])
for ix,g in enumerate(sorted(set(groups)),1):
 print(ix,g,flush=True); te=np.where(groups==g)[0];tr=np.where(groups!=g)[0]
 sadt=StandardScaler(); At=sadt.fit_transform(ADT[tr]); Av=sadt.transform(ADT[te]); pca=PCA(n_components=30,svd_solver='randomized',random_state=20260807+ix);Pt=pca.fit_transform(At);Pv=pca.transform(Av)
 pred('ADT PCA (30)',Pt,Pv,tr,te); pred('ADT PCA + classical',np.c_[Pt,cl[tr]],np.c_[Pv,cl[te]],tr,te); pred('ADT PCA + APG',np.c_[Pt,apg[tr]],np.c_[Pv,apg[te]],tr,te); pred('ADT PCA + classical + APG',np.c_[Pt,cl[tr],apg[tr]],np.c_[Pv,cl[te],apg[te]],tr,te)
 pred('RNA PCA (50)',rna[tr],rna[te],tr,te); pred('RNA PCA + ADT APG',np.c_[rna[tr],apg[tr]],np.c_[rna[te],apg[te]],tr,te)
r=pd.DataFrame(rows,columns=['group','n_test','model','balanced_accuracy','macro_f1']);r.to_csv(out/'azimuth_strict_group_metrics.csv',index=False)
s=r.groupby('model').agg(n_groups=('group','nunique'),balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index().sort_values('macro_f1_mean',ascending=False);s.to_csv(out/'azimuth_strict_group_summary.csv',index=False);print(s.to_string(index=False))
pts=[]
for a,b in [('ADT PCA + APG','ADT PCA (30)'),('ADT PCA + classical + APG','ADT PCA + classical'),('RNA PCA + ADT APG','RNA PCA (50)')]:
 da=r[r.model==a].set_index('group').macro_f1;db=r[r.model==b].set_index('group').macro_f1;d=(da-db).dropna();st,p=wilcoxon(d);pts.append([a,b,len(d),d.mean(),np.median(d),st,p])
pd.DataFrame(pts,columns=['model_a','model_b','n_groups','mean_delta_macro_f1','median_delta_macro_f1','wilcoxon_stat','p']).to_csv(out/'azimuth_strict_group_paired.csv',index=False)
