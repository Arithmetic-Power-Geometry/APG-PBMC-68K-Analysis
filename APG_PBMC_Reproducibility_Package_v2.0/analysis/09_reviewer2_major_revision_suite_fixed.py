import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import time, tracemalloc, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.kernel_approximation import RBFSampler

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'source_data'/'Fresh68k_cell_level_descriptors_and_annotations.csv.gz'
OUT=BASE/'results'; OUT.mkdir(exist_ok=True)

df=pd.read_csv(SRC)
df=df[df.cell_type!='Uncertain'].copy()
bal=df.groupby('cell_type',group_keys=False).sample(n=800,random_state=42).reset_index(drop=True)
y=bal.cell_type.to_numpy(); classes=np.sort(pd.unique(y))

gc=['shannon','renyi2','simpson','hhi','participation','gini','detected']
ga=['apg_entropy_norm','apg_cmax','apg_c2','apg_neff','apg_pr','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6']
pc=['program_shannon','program_renyi2','program_simpson','program_hhi','program_participation','program_gini','program_detected']
pa=['program_apg_entropy_norm','program_apg_cmax','program_apg_c2','program_apg_neff','program_apg_pr','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0','program_apg_integrated_2_6']
all_classical=gc+pc; all_apg=ga+pa
splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260807).split(np.zeros(len(y)),y))

def fit_predict(X,tr,te,C=1.0):
    sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[te])
    m=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=C))
    m.fit(a,y[tr]); return m.predict(b),m,sc,a,b

# 1 feature dictionary
rows=[]
for block,cols in [('gene_classical',gc),('gene_apg',ga),('programme_classical',pc),('programme_apg',pa)]:
    for c in cols:
        if 'entropy' in c: fam='entropy'
        elif 'd_' in c or 'integrated' in c: fam='exponent_deformation'
        elif 'cmax' in c: fam='maximum_concentration'
        elif c.endswith('_c2') or c=='apg_c2': fam='quadratic_concentration'
        elif 'neff' in c: fam='effective_dimension'
        elif c.endswith('_pr') or c=='apg_pr': fam='participation_ratio'
        elif 'gini' in c: fam='inequality'
        elif 'detected' in c: fam='support_size'
        else: fam='classical_concentration_diversity'
        rows.append([block,c,fam])
pd.DataFrame(rows,columns=['block','feature','mathematical_family']).to_csv(OUT/'reviewer2_feature_dictionary.csv',index=False)

# 2 stability audit
rows=[]
for c in all_classical+all_apg:
    x=pd.to_numeric(df[c],errors='coerce').to_numpy(float)
    rows.append([c,np.mean(np.isfinite(x)),np.nanmin(x),np.nanmax(x),np.nanstd(x),np.mean(x==0)])
pd.DataFrame(rows,columns=['feature','finite_fraction','min','max','sd','zero_fraction']).to_csv(OUT/'reviewer2_descriptor_stability_audit.csv',index=False)

# 3 APG correlations
corr=bal[all_apg].corr(method='spearman'); corr.to_csv(OUT/'reviewer2_apg_spearman_correlation.csv')
pairs=[]
for i,a in enumerate(all_apg):
    for b in all_apg[i+1:]: pairs.append([a,b,corr.loc[a,b],abs(corr.loc[a,b])])
pd.DataFrame(pairs,columns=['feature_a','feature_b','spearman_rho','abs_rho']).sort_values('abs_rho',ascending=False).to_csv(OUT/'reviewer2_apg_correlation_pairs.csv',index=False)

# 4 exponent sensitivity
sens={
 'Classical only':all_classical,
 'Classical + APG low-p':all_classical+['apg_entropy_norm','apg_d_2_25','apg_d_2_5','program_apg_entropy_norm','program_apg_d_2_25','program_apg_d_2_5'],
 'Classical + APG transition':all_classical+['apg_entropy_norm','apg_d_2_25','apg_d_2_5','apg_d_3_0','program_apg_entropy_norm','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0'],
 'Classical + APG high-p':all_classical+['apg_d_4_0','apg_d_5_0','apg_d_6_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0'],
 'Classical + all APG':all_classical+all_apg}
rows=[]
for name,cols in sens.items():
    X=bal[cols].to_numpy(float)
    for fold,(tr,te) in enumerate(splits,1):
        pr,_,_,_,_=fit_predict(X,tr,te)
        rows.append([name,fold,len(cols),balanced_accuracy_score(y[te],pr),f1_score(y[te],pr,average='macro')])
met=pd.DataFrame(rows,columns=['model','fold','n_features','balanced_accuracy','macro_f1']); met.to_csv(OUT/'reviewer2_exponent_sensitivity_fold_metrics.csv',index=False)
met.groupby(['model','n_features'],as_index=False).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).to_csv(OUT/'reviewer2_exponent_sensitivity_summary.csv',index=False)

# 5 nested regularized matched controls. All transformations are fit/constructed inside outer folds.
Xc=bal[all_classical].to_numpy(float); Xa=bal[all_classical+all_apg].to_numpy(float)
Cs=[0.01,0.1,1.0,10.0]; rows=[]
for fold,(tr,te) in enumerate(splits,1):
    inner=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=1000+fold).split(np.zeros(len(tr)),y[tr]))
    def build(kind,train_idx,test_idx):
        base_tr=Xc[train_idx]; base_te=Xc[test_idx]
        if kind=='classical': return base_tr,base_te
        if kind=='apg': return Xa[train_idx],Xa[test_idx]
        if kind=='polylog':
            extra_tr=np.c_[base_tr**2,np.log1p(np.maximum(base_tr[:,:10],0))]
            extra_te=np.c_[base_te**2,np.log1p(np.maximum(base_te[:,:10],0))]
            return np.c_[base_tr,extra_tr],np.c_[base_te,extra_te]
        # RFF: scaler and RFF are fit only on the supplied training partition
        pre=StandardScaler(); ztr=pre.fit_transform(base_tr); zte=pre.transform(base_te)
        rff=RBFSampler(gamma=1/14,n_components=24,random_state=20260807+fold)
        return np.c_[base_tr,rff.fit_transform(ztr)],np.c_[base_te,rff.transform(zte)]
    for label,kind,nfeat in [('All classical (14)','classical',14),('All classical + matched polynomial/log control (38)','polylog',38),('All classical + matched RFF control (38)','rff',38),('All classical + all APG (38)','apg',38)]:
        best=(-1,None)
        for C in Cs:
            ss=[]
            for itr,iva in inner:
                ti=tr[itr]; vi=tr[iva]; A,B=build(kind,ti,vi)
                sc=StandardScaler(); aa=sc.fit_transform(A); bb=sc.transform(B)
                m=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=C)); m.fit(aa,y[ti]); ss.append(f1_score(y[vi],m.predict(bb),average='macro'))
            if np.mean(ss)>best[0]: best=(np.mean(ss),C)
        A,B=build(kind,tr,te); sc=StandardScaler(); aa=sc.fit_transform(A); bb=sc.transform(B)
        m=OneVsRestClassifier(LogisticRegression(solver='liblinear',max_iter=400,class_weight='balanced',C=best[1])); m.fit(aa,y[tr]); pr=m.predict(bb)
        rows.append([label,fold,nfeat,best[1],best[0],balanced_accuracy_score(y[te],pr),f1_score(y[te],pr,average='macro')])
controls=pd.DataFrame(rows,columns=['model','fold','n_features','selected_C','inner_cv_macro_f1','balanced_accuracy','macro_f1']); controls.to_csv(OUT/'reviewer2_nested_regularized_controls_fold_metrics.csv',index=False)
controls.groupby(['model','n_features'],as_index=False).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).to_csv(OUT/'reviewer2_nested_regularized_controls_summary.csv',index=False)

# 6 per-cell-type errors/confusions
error_models={'All classical (14)':all_classical,'All classical + all APG (38)':all_classical+all_apg}; per=[]
for name,cols in error_models.items():
    X=bal[cols].to_numpy(float); oof=np.empty(len(y),dtype=object)
    for tr,te in splits: oof[te]=fit_predict(X,tr,te)[0]
    rep=classification_report(y,oof,labels=classes,output_dict=True,zero_division=0)
    for c in classes: per.append([name,c,rep[c]['precision'],rep[c]['recall'],rep[c]['f1-score'],rep[c]['support']])
    safe=name.lower().replace(' ','_').replace('+','plus').replace('(','').replace(')','')
    pd.DataFrame(confusion_matrix(y,oof,labels=classes),index=classes,columns=classes).to_csv(OUT/f'reviewer2_confusion_{safe}.csv')
pd.DataFrame(per,columns=['model','cell_type','precision','recall','f1','support']).to_csv(OUT/'reviewer2_per_cell_type_metrics.csv',index=False)

# 7 coefficients and held-out permutation importance
X=bal[all_classical+all_apg].to_numpy(float); co=[]; pirows=[]
for fold,(tr,te) in enumerate(splits,1):
    pr,m,sc,a,b=fit_predict(X,tr,te)
    for c,est in zip(m.classes_,m.estimators_):
        for feat,val in zip(all_classical+all_apg,est.coef_[0]): co.append([fold,c,feat,val,abs(val)])
    pi=permutation_importance(m,b,y[te],n_repeats=3,random_state=20260807+fold,scoring='f1_macro',n_jobs=1)
    for feat,mu,sd in zip(all_classical+all_apg,pi.importances_mean,pi.importances_std): pirows.append([fold,feat,mu,sd])
pd.DataFrame(co,columns=['fold','cell_type','feature','coefficient','abs_coefficient']).to_csv(OUT/'reviewer2_feature_coefficients.csv',index=False)
pd.DataFrame(pirows,columns=['fold','feature','permutation_importance_mean','permutation_importance_sd']).to_csv(OUT/'reviewer2_permutation_importance.csv',index=False)

# 8 learning curves using stratified subsamples inside each training fold
rows=[]
for name,cols in error_models.items():
    X=bal[cols].to_numpy(float)
    for fold,(tr,te) in enumerate(splits,1):
        td=pd.DataFrame({'idx':tr,'label':y[tr]})
        for frac in [0.2,0.4,0.6,0.8,1.0]:
            sub=td if frac==1 else td.groupby('label',group_keys=False).sample(frac=frac,random_state=20260807+fold)
            sti=sub.idx.to_numpy(); pr=fit_predict(X,sti,te)[0]
            rows.append([name,fold,frac,len(sti),f1_score(y[te],pr,average='macro')])
pd.DataFrame(rows,columns=['model','fold','train_fraction','n_train','macro_f1']).to_csv(OUT/'reviewer2_learning_curves.csv',index=False)

# 9 cost/memory
rows=[]
for name,cols in error_models.items():
    X=bal[cols].to_numpy(float); tr,te=splits[0]; tracemalloc.start(); t=time.perf_counter(); pr=fit_predict(X,tr,te)[0]; elapsed=time.perf_counter()-t; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    rows.append([name,len(cols),elapsed,peak/1024/1024,f1_score(y[te],pr,average='macro')])
pd.DataFrame(rows,columns=['model','n_features','fit_predict_seconds_fold1','peak_python_memory_mb','macro_f1_fold1']).to_csv(OUT/'reviewer2_computational_cost.csv',index=False)

# 10 controlled simulations
def desc(z):
    z=np.maximum(np.asarray(z,float),0); z=z if z.sum()>0 else z+1e-12; q=z/z.sum(); w=z*z; w=w/w.sum(); eps=1e-15
    classical=[-np.sum(q*np.log(q+eps)),-np.log(np.sum(q*q)+eps),1-np.sum(q*q),np.sum(q*q),1/(np.sum(q*q)+eps)]
    ap=[-np.sum(w*np.log(w+eps))/np.log(len(w)),np.max(w),np.sum(w*w),np.exp(-np.sum(w*np.log(w+eps))),1/np.sum(w*w)]
    ap += [1-np.sum(w**(p/2)) for p in [2.25,2.5,3,4,5,6]]
    ps=np.array([2,2.25,2.5,3,4,5,6.]); ds=np.array([1-np.sum(w**(p/2)) for p in ps]); ap.append(np.trapezoid(ds,ps))
    return classical,ap
rng=np.random.default_rng(20260807); sim=[]
for factor,levels in [('sparsity',[0,0.25,0.5,0.75]),('dispersion',[0.25,0.5,1,2]),('tail_weight',[1.5,2,4,8]),('multimodality',[0,0.25,0.5,0.75])]:
    for level in levels:
        for rep in range(200):
            m=128
            if factor=='sparsity':
                z=rng.gamma(2,1,m); z[rng.random(m)<level]=0
            elif factor=='dispersion': z=rng.gamma(shape=max(level,0.05),scale=1,size=m)
            elif factor=='tail_weight': z=rng.pareto(level,m)+1
            else:
                z=rng.gamma(2,1,m); k=int(level*m); z[:k]*=5; rng.shuffle(z)
            cl,ap=desc(z); sim.append([factor,level,rep]+cl+ap)
cols=['factor','level','rep','shannon','renyi2','simpson','hhi','participation']+ga
pd.DataFrame(sim,columns=cols).to_csv(OUT/'reviewer2_controlled_simulations.csv',index=False)

manifest={'R2_1':['reviewer2_feature_dictionary.csv','reviewer2_descriptor_stability_audit.csv','reviewer2_controlled_simulations.csv','reviewer2_apg_spearman_correlation.csv'],'R2_2':['reviewer2_feature_coefficients.csv','reviewer2_permutation_importance.csv','reviewer2_per_cell_type_metrics.csv'],'R2_3':['reviewer2_nested_regularized_controls_summary.csv'],'R2_4':['reviewer2_exponent_sensitivity_summary.csv','reviewer2_apg_correlation_pairs.csv'],'R2_5':['reviewer2_exponent_sensitivity_fold_metrics.csv','reviewer2_learning_curves.csv'],'R2_6':['reviewer2_per_cell_type_metrics.csv'],'R2_7':['CLAIM_NARROWING_REQUIRED_IF_NO_INDEPENDENT_RNA_MATRIX'],'R2_8':['reviewer2_nested_regularized_controls_summary.csv','reviewer2_computational_cost.csv'],'R2_9':['reviewer2_per_cell_type_metrics.csv','reviewer2_learning_curves.csv']}
with open(OUT/'reviewer2_coverage_manifest.json','w') as f: json.dump(manifest,f,indent=2)
print('Reviewer 2 fixed major-revision suite complete.')
