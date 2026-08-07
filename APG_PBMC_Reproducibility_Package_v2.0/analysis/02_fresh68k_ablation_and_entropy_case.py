import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import wilcoxon, spearmanr
BASE=Path(__file__).resolve().parents[1]; out=BASE/'results'; out.mkdir(exist_ok=True)
df=pd.read_csv(BASE/'source_data'/'Fresh68k_cell_level_descriptors_and_annotations.csv.gz'); df=df[df.cell_type!='Uncertain'].copy()
bal=df.groupby('cell_type',group_keys=False).sample(n=800,random_state=42); y=bal.cell_type.to_numpy()
gc=['shannon','renyi2','simpson','hhi','participation','gini','detected']; ga=['apg_entropy_norm','apg_cmax','apg_c2','apg_neff','apg_pr','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6']; pc=['program_shannon','program_renyi2','program_simpson','program_hhi','program_participation','program_gini','program_detected']; pa=['program_apg_entropy_norm','program_apg_cmax','program_apg_c2','program_apg_neff','program_apg_pr','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0','program_apg_integrated_2_6']
mods={'Gene classical':gc,'Gene classical + APG':gc+ga,'Programme classical':pc,'Programme APG':pa,'Programme classical + APG':pc+pa,'All classical':gc+pc,'All classical + all APG':gc+pc+ga+pa}
splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260807).split(np.zeros(len(y)),y)); rows=[]
for name,cols in mods.items():
 X=bal[cols].to_numpy(float)
 for f,(tr,te) in enumerate(splits,1):
  sc=StandardScaler(); xt=sc.fit_transform(X[tr]); xv=sc.transform(X[te]); m=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=300,class_weight='balanced',C=1.0)); m.fit(xt,y[tr]); pr=m.predict(xv); rows.append([name,f,balanced_accuracy_score(y[te],pr),f1_score(y[te],pr,average='macro')])
r=pd.DataFrame(rows,columns=['model','fold','balanced_accuracy','macro_f1']); r.to_csv(out/'fresh68k_ablation_fold_metrics.csv',index=False)
s=r.groupby('model').agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index(); s['features']=s.model.map({k:len(v) for k,v in mods.items()}); s=s.sort_values('macro_f1_mean',ascending=False); s.to_csv(out/'fresh68k_ablation_summary.csv',index=False)
pts=[]
for a,b in [('Programme classical + APG','Programme classical'),('All classical + all APG','All classical'),('Gene classical + APG','Gene classical')]:
 da=r[r.model==a].sort_values('fold').macro_f1.to_numpy(); db=r[r.model==b].sort_values('fold').macro_f1.to_numpy(); d=da-db; st,p=wilcoxon(d); pts.append([a,b,len(d),d.mean(),np.median(d),st,p])
pd.DataFrame(pts,columns=['model_a','model_b','n_folds','mean_delta_macro_f1','median_delta_macro_f1','wilcoxon_stat','p']).to_csv(out/'fresh68k_ablation_paired_tests.csv',index=False)
# Redundancy relative to APG entropy
x=df[['apg_entropy_norm']].to_numpy(); red=[]
for c in ['apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6']:
 yy=df[c].to_numpy(); rho,_=spearmanr(x[:,0],yy); r2=LinearRegression().fit(x,yy).score(x,yy); red.append([c,rho,r2,yy.std()])
pd.DataFrame(red,columns=['descriptor','spearman_with_apg_entropy_norm','linear_r2_from_apg_entropy_norm','sd']).to_csv(out/'fresh68k_entropy_redundancy.csv',index=False)
# Ordinary-Shannon-matched biological illustration: within a single provisional class, find
# a pair with Shannon difference <=0.0011 that maximizes the D(3) contrast.
best=None
for cell_type, sub in df.groupby('cell_type'):
 sub=sub.sort_values('shannon').reset_index(drop=True); vals=sub.shannon.to_numpy(); d3=sub.apg_d_3_0.to_numpy(); tol=5e-4
 for i in range(len(sub)):
  lo=np.searchsorted(vals,vals[i]-tol); hi=np.searchsorted(vals,vals[i]+tol,side='right')
  if hi-lo<2: continue
  j=lo+np.nanargmax(d3[lo:hi]); k=lo+np.nanargmin(d3[lo:hi]); diff=abs(vals[j]-vals[k]); gap=d3[j]-d3[k]
  if diff<=0.0011 and (best is None or gap>best[0]): best=(gap,cell_type,j,k,sub)
if best:
 gap,cell_type,j,k,sub=best; cols=['barcode','cell_type','shannon','apg_entropy_norm','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_integrated_2_6','score_CD4_T','score_CD8_T','score_NK','score_B_cell','score_CD14_monocyte','score_FCGR3A_monocyte','score_Dendritic','score_Platelet']; sub.iloc[[j,k]][cols].to_csv(out/'fresh68k_entropy_matched_case.csv',index=False)
