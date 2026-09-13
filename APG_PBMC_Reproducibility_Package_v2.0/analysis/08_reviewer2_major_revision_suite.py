import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json, time, tracemalloc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'source_data'/'Fresh68k_cell_level_descriptors_and_annotations.csv.gz'
OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
FIG=BASE/'figures'; FIG.mkdir(exist_ok=True)

df=pd.read_csv(SRC)
df=df[df.cell_type!='Uncertain'].copy()
bal=df.groupby('cell_type',group_keys=False).sample(n=800,random_state=42)
y=bal.cell_type.to_numpy()
classes=np.sort(pd.unique(y))

gc=['shannon','renyi2','simpson','hhi','participation','gini','detected']
ga=['apg_entropy_norm','apg_cmax','apg_c2','apg_neff','apg_pr','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6']
pc=['program_shannon','program_renyi2','program_simpson','program_hhi','program_participation','program_gini','program_detected']
pa=['program_apg_entropy_norm','program_apg_cmax','program_apg_c2','program_apg_neff','program_apg_pr','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0','program_apg_integrated_2_6']
all_apg=ga+pa
all_classical=gc+pc

# ---------------------------
# 1. Formal feature dictionary
# ---------------------------
rows=[]
for level,cols in [('gene_classical',gc),('gene_apg',ga),('programme_classical',pc),('programme_apg',pa)]:
    for c in cols:
        if 'entropy' in c: family='entropy'
        elif 'd_' in c or 'integrated' in c: family='exponent_deformation'
        elif 'cmax' in c: family='maximum_concentration'
        elif c.endswith('_c2') or c=='apg_c2': family='quadratic_concentration'
        elif 'neff' in c: family='effective_dimension'
        elif c.endswith('_pr') or c=='apg_pr': family='participation_ratio'
        elif 'gini' in c: family='inequality'
        elif 'detected' in c: family='support_size'
        elif 'simpson' in c or 'hhi' in c or 'participation' in c or 'renyi2' in c: family='classical_concentration_diversity'
        else: family='other'
        rows.append([level,c,family])
pd.DataFrame(rows,columns=['block','feature','mathematical_family']).to_csv(OUT/'reviewer2_feature_dictionary.csv',index=False)

# ---------------------------
# 2. Scale-invariance/stability checks on derived descriptors
# ---------------------------
# Descriptors already computed from normalized profiles should be invariant to a common scale
# before normalization by construction. We record this formal property and empirical finite checks
# that can be performed from available descriptor columns (boundedness, zero/finite rates).
stab=[]
for c in all_classical+all_apg:
    x=pd.to_numeric(df[c],errors='coerce').to_numpy(float)
    stab.append([c,np.mean(np.isfinite(x)),np.nanmin(x),np.nanmax(x),np.nanstd(x),np.mean(x==0)])
pd.DataFrame(stab,columns=['feature','finite_fraction','min','max','sd','zero_fraction']).to_csv(OUT/'reviewer2_descriptor_stability_audit.csv',index=False)

# ---------------------------
# 3. APG correlation/redundancy matrix
# ---------------------------
corr=bal[all_apg].corr(method='spearman')
corr.to_csv(OUT/'reviewer2_apg_spearman_correlation.csv')
pairs=[]
for i,a in enumerate(all_apg):
    for b in all_apg[i+1:]:
        pairs.append([a,b,corr.loc[a,b],abs(corr.loc[a,b])])
pd.DataFrame(pairs,columns=['feature_a','feature_b','spearman_rho','abs_rho']).sort_values('abs_rho',ascending=False).to_csv(OUT/'reviewer2_apg_correlation_pairs.csv',index=False)

# ---------------------------
# 4. Prespecified low/moderate/high exponent sensitivity
# ---------------------------
groups={
 'Classical only':all_classical,
 'Classical + APG low-p':all_classical+['apg_entropy_norm','apg_d_2_25','apg_d_2_5','program_apg_entropy_norm','program_apg_d_2_25','program_apg_d_2_5'],
 'Classical + APG transition':all_classical+['apg_entropy_norm','apg_d_2_25','apg_d_2_5','apg_d_3_0','program_apg_entropy_norm','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0'],
 'Classical + APG high-p':all_classical+['apg_d_4_0','apg_d_5_0','apg_d_6_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0'],
 'Classical + all APG':all_classical+all_apg,
}
splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260807).split(np.zeros(len(y)),y))
metrics=[]; predictions=[]
for name,cols in groups.items():
    X=bal[cols].to_numpy(float)
    for fold,(tr,te) in enumerate(splits,1):
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
        m=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=1.0))
        m.fit(Xtr,y[tr]); pred=m.predict(Xte)
        metrics.append([name,fold,len(cols),balanced_accuracy_score(y[te],pred),f1_score(y[te],pred,average='macro')])
        for idx,p in zip(te,pred): predictions.append([name,fold,idx,y[idx],p])
met=pd.DataFrame(metrics,columns=['model','fold','n_features','balanced_accuracy','macro_f1'])
met.to_csv(OUT/'reviewer2_exponent_sensitivity_fold_metrics.csv',index=False)
met.groupby(['model','n_features'],as_index=False).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).to_csv(OUT/'reviewer2_exponent_sensitivity_summary.csv',index=False)

# ---------------------------
# 5. Stronger equal-dimensional controls + regularisation grid
# ---------------------------
# Every model here uses the same training/test folds. Hyperparameter C is selected INSIDE training
# data by an inner 3-fold CV; held-out outer folds remain untouched.
def generic_matched(frame,cols,prefix):
    x=frame[cols].to_numpy(float)
    out={}
    for j in range(len(cols)): out[f'{prefix}_sq_{j+1}']=x[:,j]**2
    for j in range(min(5,len(cols))): out[f'{prefix}_log_{j+1}']=np.log1p(np.maximum(x[:,j],0))
    return pd.DataFrame(out,index=frame.index)

gctrl=generic_matched(bal,gc,'gctrl'); pctrl=generic_matched(bal,pc,'pctrl')
for c in gctrl: bal[c]=gctrl[c]
for c in pctrl: bal[c]=pctrl[c]
ctrl_cols=list(gctrl.columns)+list(pctrl.columns)

# Random Fourier features from 14 classical inputs, exactly 24 extra coordinates -> 38 total.
rff=RBFSampler(gamma=1.0/14,n_components=24,random_state=20260807)
Xc=bal[all_classical].to_numpy(float)
# fit_transform is label-free; standardization remains training-fold-specific below.
Xrff=rff.fit_transform(StandardScaler().fit_transform(Xc))
for j in range(24): bal[f'rff_{j+1}']=Xrff[:,j]
rff_cols=[f'rff_{j+1}' for j in range(24)]

models={
 'All classical (14)':all_classical,
 'All classical + matched polynomial/log control (38)':all_classical+ctrl_cols,
 'All classical + matched RFF control (38)':all_classical+rff_cols,
 'All classical + all APG (38)':all_classical+all_apg,
}
Cs=[0.01,0.1,1.0,10.0]
control_rows=[]
for name,cols in models.items():
    X=bal[cols].to_numpy(float)
    for fold,(tr,te) in enumerate(splits,1):
        inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=1000+fold)
        best=(-np.inf,None)
        for C in Cs:
            scores=[]
            for itr,iva in inner.split(X[tr],y[tr]):
                tr2=tr[itr]; va2=tr[iva]
                sc=StandardScaler(); a=sc.fit_transform(X[tr2]); b=sc.transform(X[va2])
                mm=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=C))
                mm.fit(a,y[tr2]); scores.append(f1_score(y[va2],mm.predict(b),average='macro'))
            if np.mean(scores)>best[0]: best=(np.mean(scores),C)
        sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[te])
        mm=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=best[1]))
        mm.fit(a,y[tr]); pred=mm.predict(b)
        control_rows.append([name,fold,len(cols),best[1],best[0],balanced_accuracy_score(y[te],pred),f1_score(y[te],pred,average='macro')])
controls=pd.DataFrame(control_rows,columns=['model','fold','n_features','selected_C','inner_cv_macro_f1','balanced_accuracy','macro_f1'])
controls.to_csv(OUT/'reviewer2_nested_regularized_controls_fold_metrics.csv',index=False)
controls.groupby(['model','n_features'],as_index=False).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).to_csv(OUT/'reviewer2_nested_regularized_controls_summary.csv',index=False)

# ---------------------------
# 6. Per-cell-type error analysis and confusion matrices
# ---------------------------
# Use out-of-fold predictions for classical and APG models from fresh five-fold fits.
error_models={'All classical (14)':all_classical,'All classical + all APG (38)':all_classical+all_apg}
perclass=[]
for name,cols in error_models.items():
    X=bal[cols].to_numpy(float); oof=np.empty(len(y),dtype=object)
    for fold,(tr,te) in enumerate(splits,1):
        sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[te])
        mm=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=1.0)); mm.fit(a,y[tr]); oof[te]=mm.predict(b)
    rep=classification_report(y,oof,labels=classes,output_dict=True,zero_division=0)
    for cl in classes: perclass.append([name,cl,rep[cl]['precision'],rep[cl]['recall'],rep[cl]['f1-score'],rep[cl]['support']])
    pd.DataFrame(confusion_matrix(y,oof,labels=classes),index=classes,columns=classes).to_csv(OUT/f"reviewer2_confusion_{name.lower().replace(' ','_').replace('+','plus').replace('(','').replace(')','')}.csv")
pd.DataFrame(perclass,columns=['model','cell_type','precision','recall','f1','support']).to_csv(OUT/'reviewer2_per_cell_type_metrics.csv',index=False)

# ---------------------------
# 7. Interpretability via coefficients + permutation importance
# ---------------------------
X=bal[all_classical+all_apg].to_numpy(float)
coef_rows=[]; perm_rows=[]
for fold,(tr,te) in enumerate(splits,1):
    sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[te])
    mm=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=1.0)); mm.fit(a,y[tr])
    for cl,est in zip(mm.classes_,mm.estimators_):
        for feat,val in zip(all_classical+all_apg,est.coef_[0]): coef_rows.append([fold,cl,feat,val,abs(val)])
    # whole-estimator permutation importance on held-out fold, macro-F1 scorer
    pi=permutation_importance(mm,b,y[te],n_repeats=5,random_state=20260807+fold,scoring='f1_macro',n_jobs=1)
    for feat,mean,sd in zip(all_classical+all_apg,pi.importances_mean,pi.importances_std): perm_rows.append([fold,feat,mean,sd])
pd.DataFrame(coef_rows,columns=['fold','cell_type','feature','coefficient','abs_coefficient']).to_csv(OUT/'reviewer2_feature_coefficients.csv',index=False)
pd.DataFrame(perm_rows,columns=['fold','feature','permutation_importance_mean','permutation_importance_sd']).to_csv(OUT/'reviewer2_permutation_importance.csv',index=False)

# ---------------------------
# 8. Learning curves
# ---------------------------
learn=[]
train_fracs=[0.2,0.4,0.6,0.8,1.0]
for name,cols in error_models.items():
    X=bal[cols].to_numpy(float)
    for fold,(tr,te) in enumerate(splits,1):
        rng=np.random.default_rng(20260807+fold)
        for frac in train_fracs:
            # stratified deterministic subsample within outer training split
            td=pd.DataFrame({'idx':tr,'y':y[tr]})
            sub=td.groupby('y',group_keys=False).sample(frac=frac,random_state=20260807+fold) if frac<1 else td
            sti=sub.idx.to_numpy()
            sc=StandardScaler(); a=sc.fit_transform(X[sti]); b=sc.transform(X[te])
            mm=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=1.0)); mm.fit(a,y[sti]); pred=mm.predict(b)
            learn.append([name,fold,frac,len(sti),f1_score(y[te],pred,average='macro')])
pd.DataFrame(learn,columns=['model','fold','train_fraction','n_train','macro_f1']).to_csv(OUT/'reviewer2_learning_curves.csv',index=False)

# ---------------------------
# 9. Computational cost/memory
# ---------------------------
cost=[]
for name,cols in error_models.items():
    X=bal[cols].to_numpy(float); tr,te=splits[0]
    tracemalloc.start(); t0=time.perf_counter()
    sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[te])
    mm=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=1.0)); mm.fit(a,y[tr]); pred=mm.predict(b)
    elapsed=time.perf_counter()-t0; cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    cost.append([name,len(cols),elapsed,peak/1024/1024,f1_score(y[te],pred,average='macro')])
pd.DataFrame(cost,columns=['model','n_features','fit_predict_seconds_fold1','peak_python_memory_mb','macro_f1_fold1']).to_csv(OUT/'reviewer2_computational_cost.csv',index=False)

# ---------------------------
# 10. Controlled simulations: one-factor-at-a-time profile perturbations
# ---------------------------
def apg_features(z):
    z=np.asarray(z,float); z=np.maximum(z,0)
    if z.sum()==0: z=z+1e-12
    q=z/z.sum(); w=z*z; w=w/w.sum()
    eps=1e-15
    sh=-np.sum(q*np.log(q+eps)); h=-np.sum(w*np.log(w+eps))/np.log(len(w))
    vals=[h,np.max(w),np.sum(w*w),np.exp(-np.sum(w*np.log(w+eps))),1/np.sum(w*w)]
    for p in [2.25,2.5,3,4,5,6]: vals.append(1-np.sum(w**(p/2)))
    ps=np.array([2,2.25,2.5,3,4,5,6.]); ds=np.array([1-np.sum(w**(p/2)) for p in ps]); vals.append(np.trapz(ds,ps))
    classical=[sh,-np.log(np.sum(q*q)+eps),1-np.sum(q*q),np.sum(q*q),1/(np.sum(q*q)+eps),np.nan,np.sum(z>0)]
    return classical,vals
rng=np.random.default_rng(20260807)
sim=[]
for factor,levels in [('sparsity',[0,0.25,0.5,0.75]),('dispersion',[0.25,0.5,1,2]),('tail_weight',[1.5,2,4,8]),('multimodality',[0,0.25,0.5,0.75])]:
    for level in levels:
        for rep in range(200):
            m=128
            if factor=='sparsity':
                z=rng.gamma(2,1,m); z[rng.random(m)<level]=0
            elif factor=='dispersion': z=rng.gamma(shape=max(level,0.05),scale=1,m)
            elif factor=='tail_weight': z=(rng.pareto(level,m)+1)
            else:
                z=rng.gamma(2,1,m); k=int(level*m); z[:k]*=5; rng.shuffle(z)
            cl,ap=apg_features(z)
            sim.append([factor,level,rep]+cl[:5]+ap)
cols=['factor','level','rep','shannon','renyi2','simpson','hhi','participation']+all_apg[:12]
pd.DataFrame(sim,columns=cols).to_csv(OUT/'reviewer2_controlled_simulations.csv',index=False)

# Reviewer-response coverage manifest
manifest={
 'R2_1':['reviewer2_feature_dictionary.csv','reviewer2_descriptor_stability_audit.csv','reviewer2_controlled_simulations.csv','reviewer2_apg_spearman_correlation.csv'],
 'R2_2':['reviewer2_feature_coefficients.csv','reviewer2_permutation_importance.csv','reviewer2_per_cell_type_metrics.csv'],
 'R2_3':['reviewer2_nested_regularized_controls_summary.csv','reviewer2_nested_regularized_controls_fold_metrics.csv'],
 'R2_4':['reviewer2_exponent_sensitivity_summary.csv','reviewer2_apg_correlation_pairs.csv'],
 'R2_5':['reviewer2_exponent_sensitivity_fold_metrics.csv','reviewer2_learning_curves.csv'],
 'R2_6':['reviewer2_per_cell_type_metrics.csv'],
 'R2_7':['CLAIM_NARROWING_REQUIRED_IF_NO_INDEPENDENT_RNA_MATRIX'],
 'R2_8':['reviewer2_nested_regularized_controls_summary.csv','reviewer2_computational_cost.csv'],
 'R2_9':['reviewer2_per_cell_type_metrics.csv','reviewer2_learning_curves.csv']
}
with open(OUT/'reviewer2_coverage_manifest.json','w') as f: json.dump(manifest,f,indent=2)
print('Reviewer 2 major-revision suite complete.')
