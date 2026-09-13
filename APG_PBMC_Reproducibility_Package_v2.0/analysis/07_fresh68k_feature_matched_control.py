import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'source_data'/'Fresh68k_cell_level_descriptors_and_annotations.csv.gz'
OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(SRC)
df=df[df.cell_type!='Uncertain'].copy()
bal=df.groupby('cell_type',group_keys=False).sample(n=800,random_state=42)
y=bal.cell_type.to_numpy()

gc=['shannon','renyi2','simpson','hhi','participation','gini','detected']
ga=['apg_entropy_norm','apg_cmax','apg_c2','apg_neff','apg_pr','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6']
pc=['program_shannon','program_renyi2','program_simpson','program_hhi','program_participation','program_gini','program_detected']
pa=['program_apg_entropy_norm','program_apg_cmax','program_apg_c2','program_apg_neff','program_apg_pr','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0','program_apg_integrated_2_6']

# Twelve deterministic, label-independent nonlinear controls per 7-variable
# classical block: seven squared terms plus five log1p terms. This creates
# exactly the same number of added coordinates as the 12 APG descriptors.
# The transforms are intentionally generic and are computed from the same
# classical input summaries; no labels are used in their construction.
def nonlinear_bank(frame, cols, prefix):
    x=frame[cols].to_numpy(float)
    out={}
    for j,c in enumerate(cols):
        out[f'{prefix}_sq_{j+1}']=x[:,j]**2
    for j,c in enumerate(cols[:5]):
        out[f'{prefix}_log_{j+1}']=np.log1p(np.maximum(x[:,j],0.0))
    z=pd.DataFrame(out,index=frame.index)
    assert z.shape[1]==12
    return z

gctrl=nonlinear_bank(bal,gc,'gene_ctrl')
pctrl=nonlinear_bank(bal,pc,'program_ctrl')
for c in gctrl.columns: bal[c]=gctrl[c]
for c in pctrl.columns: bal[c]=pctrl[c]
gc_ctrl=list(gctrl.columns); pc_ctrl=list(pctrl.columns)

mods={
 'Gene classical (7)':gc,
 'Gene classical + matched nonlinear control (19)':gc+gc_ctrl,
 'Gene classical + APG (19)':gc+ga,
 'Programme classical (7)':pc,
 'Programme classical + matched nonlinear control (19)':pc+pc_ctrl,
 'Programme classical + APG (19)':pc+pa,
 'All classical (14)':gc+pc,
 'All classical + matched nonlinear control (38)':gc+pc+gc_ctrl+pc_ctrl,
 'All classical + all APG (38)':gc+pc+ga+pa,
}

splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260807).split(np.zeros(len(y)),y))
rows=[]
for name,cols in mods.items():
    X=bal[cols].to_numpy(float)
    for fold,(tr,te) in enumerate(splits,1):
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
        m=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=300,class_weight='balanced',C=1.0))
        m.fit(Xtr,y[tr]); pred=m.predict(Xte)
        rows.append({'model':name,'fold':fold,'n_features':len(cols),
                     'balanced_accuracy':balanced_accuracy_score(y[te],pred),
                     'macro_f1':f1_score(y[te],pred,average='macro')})

r=pd.DataFrame(rows); r.to_csv(OUT/'fresh68k_feature_matched_metrics.csv',index=False)
s=r.groupby(['model','n_features'],as_index=False).agg(
    balanced_accuracy_mean=('balanced_accuracy','mean'),
    balanced_accuracy_sd=('balanced_accuracy','std'),
    macro_f1_mean=('macro_f1','mean'),
    macro_f1_sd=('macro_f1','std'))
s.to_csv(OUT/'fresh68k_feature_matched_summary.csv',index=False)

pairs=[
 ('Gene_APG_vs_matched','Gene classical + APG (19)','Gene classical + matched nonlinear control (19)'),
 ('Programme_APG_vs_matched','Programme classical + APG (19)','Programme classical + matched nonlinear control (19)'),
 ('All_APG_vs_matched','All classical + all APG (38)','All classical + matched nonlinear control (38)')]
paired=[]
for label,a_name,b_name in pairs:
    a=r[r.model==a_name].sort_values('fold').macro_f1.to_numpy()
    b=r[r.model==b_name].sort_values('fold').macro_f1.to_numpy()
    d=a-b; mean=d.mean(); sem=stats.sem(d)
    ci=stats.t.interval(.95,len(d)-1,loc=mean,scale=sem) if sem>0 else (mean,mean)
    w=stats.wilcoxon(d,alternative='two-sided',zero_method='wilcox')
    paired.append({'comparison':label,'mean_macro_f1_difference_apg_minus_control':mean,
                   'ci95_low_fold_level_t':ci[0],'ci95_high_fold_level_t':ci[1],
                   'wilcoxon_statistic':w.statistic,'wilcoxon_p_two_sided':w.pvalue,
                   'all_folds_apg_higher':bool(np.all(d>0)),
                   'fold_differences':';'.join(f'{x:.8f}' for x in d)})
pd.DataFrame(paired).to_csv(OUT/'fresh68k_feature_matched_paired.csv',index=False)
print(s.to_string(index=False))
print(pd.DataFrame(paired).to_string(index=False))
