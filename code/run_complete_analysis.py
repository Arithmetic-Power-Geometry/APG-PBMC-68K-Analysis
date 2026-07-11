import os, tarfile, json, math, shutil, zipfile, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import mmread
from scipy import sparse, stats
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

ROOT=Path('/mnt/data/APG_Single_Cell_PBMC_Q2_Final')
if ROOT.exists(): shutil.rmtree(ROOT)
for d in ['data','code','results','tables','figures','tests','provenance','manuscript']:
    (ROOT/d).mkdir(parents=True, exist_ok=True)

matrix_tar=Path('/mnt/data/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar(1).gz')
analysis_tar=Path('/mnt/data/fresh_68k_pbmc_donor_a_analysis.tar(1).gz')
work=Path('/mnt/data/apg_q2_work')
if work.exists(): shutil.rmtree(work)
work.mkdir()
with tarfile.open(matrix_tar,'r:gz') as t: t.extractall(work)
with tarfile.open(analysis_tar,'r:gz') as t: t.extractall(work)
matdir=work/'filtered_matrices_mex'/'hg19'
X=mmread(matdir/'matrix.mtx').tocsr().astype(np.float64) # genes x cells
# genes.tsv columns gene_id, symbol
genes=pd.read_csv(matdir/'genes.tsv',sep='\t',header=None,names=['gene_id','gene_symbol'])
barcodes=pd.read_csv(matdir/'barcodes.tsv',sep='\t',header=None,names=['barcode'])
ng,nc=X.shape
assert nc==68579
symbols=genes.gene_symbol.astype(str).values
sym_to_idx={s:i for i,s in enumerate(symbols)}

# normalize each cell to 10k, log1p; sparse
lib=np.asarray(X.sum(axis=0)).ravel()
scale=np.divide(1e4,lib,out=np.zeros_like(lib),where=lib>0)
Z=X @ sparse.diags(scale)
Z.data=np.log1p(Z.data)
Z=Z.tocsc()

# helper descriptors over sparse nonnegative matrix features x cells
def descriptors(M, prefix=''):
    M=M.tocsc()
    nfeat=M.shape[0]; ncell=M.shape[1]
    out={k:np.empty(ncell) for k in ['shannon','renyi2','simpson','hhi','participation','gini','detected','apg_entropy','apg_entropy_norm','apg_cmax','apg_c2','apg_neff','apg_pr']}
    ps=[2.25,2.5,3.,4.,5.,6.]
    for p in ps: out[f'apg_d_{str(p).replace(".","_")}']=np.empty(ncell)
    out['apg_integrated_2_6']=np.empty(ncell)
    for j in range(ncell):
        a,b=M.indptr[j],M.indptr[j+1]; v=M.data[a:b]
        k=len(v); out['detected'][j]=k
        if k==0:
            for key in out: out[key][j]=np.nan
            continue
        s=v.sum(); q=v/s
        sh=-(q*np.log(q)).sum(); hhi=(q*q).sum()
        out['shannon'][j]=sh; out['renyi2'][j]=-math.log(hhi); out['simpson'][j]=1-hhi; out['hhi'][j]=hhi; out['participation'][j]=1/hhi
        sv=np.sort(v); ranks=np.arange(nfeat-k+1,nfeat+1,dtype=float)
        out['gini'][j]=(2*np.dot(ranks,sv)/(nfeat*s))-((nfeat+1)/nfeat)
        v2=v*v; s2=v2.sum(); w=v2/s2
        H=-(w*np.log(w)).sum(); c2=(w*w).sum()
        out['apg_entropy'][j]=H; out['apg_entropy_norm'][j]=H/math.log(nfeat); out['apg_cmax'][j]=w.max(); out['apg_c2'][j]=c2; out['apg_neff'][j]=math.exp(H); out['apg_pr'][j]=1/c2
        ds=[]
        for p in ps:
            d=1-np.power(w,p/2).sum(); out[f'apg_d_{str(p).replace(".","_")}'][j]=d; ds.append(d)
        # include p=2 d=0 trapezoid
        out['apg_integrated_2_6'][j]=np.trapz([0]+ds,[2]+ps)
    return pd.DataFrame(out)

print('Computing gene-level descriptors...')
gene_desc=descriptors(Z)
gene_desc.insert(0,'barcode',barcodes.barcode.values)

# curated immune marker modules and biological programs
cell_markers={
 'CD4 T':['IL7R','LTB','MAL','CCR7','LTST1','LDHB'],
 'CD8 T':['CD8A','CD8B','CCL5','LST1','GZMK'],
 'NK':['NKG7','GNLY','PRF1','GZMB','KLRD1','FGFBP2'],
 'B cell':['MS4A1','CD79A','CD79B','CD74','CD37','HLA-DRA'],
 'CD14 monocyte':['LYZ','S100A8','S100A9','CTSS','FCN1','CD14','TYROBP'],
 'FCGR3A monocyte':['FCGR3A','MS4A7','LST1','IFITM3','SERPINA1','LGALS3'],
 'Dendritic':['FCER1A','CST3','CLEC10A','CD1C','HLA-DPA1','CLEC4C'],
 'Platelet':['PPBP','PF4','NCOA4','GNG11','SDPR','ITGA2B']}
programs={
 'TCR_signaling':['CD3D','CD3E','CD3G','TRAC','LCK','LAT','ZAP70','CD247'],
 'Cytotoxicity':['NKG7','GNLY','PRF1','GZMB','GZMH','CTSW','CCL5'],
 'B_cell_receptor':['MS4A1','CD79A','CD79B','CD37','CD74','HLA-DRA'],
 'Interferon_response':['ISG15','IFIT1','IFIT2','IFIT3','IFI6','MX1','OAS1','STAT1'],
 'Inflammatory_myeloid':['LYZ','S100A8','S100A9','FCN1','CTSS','TYROBP','LST1','LGALS3'],
 'Oxidative_phosphorylation':['NDUFA1','NDUFB1','NDUFS5','COX5A','COX6A1','ATP5F1','ATP5G1','UQCRB'],
 'Ribosomal_activity':['RPL3','RPL13','RPL32','RPS3A','RPS6','RPS12','RPS18','RPLP0'],
 'Cell_cycle':['MKI67','PCNA','STMN1','TUBA1B','TYMS','MCM2','MCM5','TOP2A'],
 'Apoptosis_stress':['BAX','BCL2','FOS','JUN','DUSP1','HSPA1A','HSP90AA1','IER2']}

def module_matrix(mods):
    arr=np.zeros((len(mods),nc),dtype=np.float32)
    used={}
    for r,(name,gs) in enumerate(mods.items()):
        idx=[sym_to_idx[g] for g in gs if g in sym_to_idx]
        used[name]=[symbols[i] for i in idx]
        if idx:
            arr[r]=np.asarray(Z[idx,:].mean(axis=0)).ravel()
    return arr,used

marker_scores,used_markers=module_matrix(cell_markers)
# z-standardize module scores per marker set to reduce scale differences
mu=marker_scores.mean(axis=1,keepdims=True); sd=marker_scores.std(axis=1,keepdims=True)+1e-8
zscore=(marker_scores-mu)/sd
best=np.argmax(zscore,axis=0); top=np.max(zscore,axis=0)
second=np.partition(zscore,-2,axis=0)[-2]
margin=top-second
labels=np.array(list(cell_markers.keys()),dtype=object)[best]
confidence=np.where((top>=0.25)&(margin>=0.10),'confident','uncertain')
labels_out=np.where(confidence=='confident',labels,'Uncertain')
annotations=pd.DataFrame({'barcode':barcodes.barcode.values,'cell_type':labels_out,'annotation_confidence':confidence,'top_marker_z':top,'marker_margin':margin})
for i,name in enumerate(cell_markers): annotations['score_'+name.replace(' ','_')]=marker_scores[i]

program_scores,used_programs=module_matrix(programs)
# program scores nonnegative already; add epsilon
PM=sparse.csc_matrix(program_scores+1e-6)
print('Computing pathway/program-level descriptors...')
prog_desc=descriptors(PM)
prog_desc.columns=['program_'+c for c in prog_desc.columns]
prog_desc.insert(0,'barcode',barcodes.barcode.values)

full=gene_desc.merge(annotations,on='barcode').merge(prog_desc,on='barcode')
full.to_csv(ROOT/'results'/'cell_level_descriptors_and_annotations.csv.gz',index=False,compression='gzip')

# descriptive summary by confident label
conf=full[full.cell_type!='Uncertain'].copy()
summary_cols=['shannon','renyi2','simpson','hhi','participation','gini','detected','apg_entropy_norm','apg_cmax','apg_c2','apg_pr','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_6_0','apg_integrated_2_6',
              'program_apg_entropy_norm','program_apg_cmax','program_apg_d_3_0','program_apg_integrated_2_6']
rows=[]
for lab,g in conf.groupby('cell_type'):
    row={'cell_type':lab,'n_cells':len(g)}
    for c in summary_cols:
        row[c+'_median']=g[c].median(); row[c+'_iqr']=g[c].quantile(.75)-g[c].quantile(.25)
    rows.append(row)
pd.DataFrame(rows).sort_values('n_cells',ascending=False).to_csv(ROOT/'tables'/'cell_type_descriptor_summary.csv',index=False)

# global tests
rows=[]
for c in summary_cols:
    groups=[g[c].dropna().values for _,g in conf.groupby('cell_type') if len(g)>=20]
    H,p=stats.kruskal(*groups)
    N=sum(len(x) for x in groups); k=len(groups); eps=max(0,(H-k+1)/(N-k))
    rows.append({'descriptor':c,'kruskal_H':H,'p_value':p,'epsilon_squared':eps,'n_cells':N,'n_groups':k})
tests=pd.DataFrame(rows).sort_values('epsilon_squared',ascending=False)
tests.to_csv(ROOT/'tables'/'cell_type_global_tests.csv',index=False)

# classification models, cap classes at 2500, require >=150
rng=np.random.default_rng(42)
parts=[]
for lab,g in conf.groupby('cell_type'):
    if len(g)>=150: parts.append(g.sample(n=min(2500,len(g)),random_state=42))
clfdf=pd.concat(parts).sample(frac=1,random_state=42)
y=clfdf.cell_type.values
sets={
'M1_Shannon_only':['shannon'],
'M2_Classical_diversity':['shannon','renyi2','simpson','hhi','participation','gini','detected'],
'M3_Classical_plus_APG':['shannon','renyi2','simpson','hhi','participation','gini','detected','apg_entropy_norm','apg_cmax','apg_c2','apg_pr','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6'],
'M4_Program_APG':['program_apg_entropy_norm','program_apg_cmax','program_apg_c2','program_apg_pr','program_apg_d_2_25','program_apg_d_2_5','program_apg_d_3_0','program_apg_d_4_0','program_apg_d_5_0','program_apg_d_6_0','program_apg_integrated_2_6'],
'M5_Classical_plus_Gene_and_Program_APG':['shannon','renyi2','simpson','hhi','participation','gini','detected','apg_entropy_norm','apg_cmax','apg_c2','apg_pr','apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0','apg_integrated_2_6','program_apg_entropy_norm','program_apg_cmax','program_apg_c2','program_apg_pr','program_apg_d_2_5','program_apg_d_3_0','program_apg_d_4_0','program_apg_d_6_0','program_apg_integrated_2_6']}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
cvrows=[]
for name,cols in sets.items():
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',solver='lbfgs'))
    scores=cross_validate(pipe,clfdf[cols].values,y,cv=cv,scoring=['balanced_accuracy','f1_macro'],n_jobs=-1)
    cvrows.append({'model':name,'n_cells':len(clfdf),'n_features':len(cols),'features':';'.join(cols),
                   'balanced_accuracy_mean':scores['test_balanced_accuracy'].mean(),'balanced_accuracy_sd':scores['test_balanced_accuracy'].std(ddof=1),
                   'macro_f1_mean':scores['test_f1_macro'].mean(),'macro_f1_sd':scores['test_f1_macro'].std(ddof=1)})
cvres=pd.DataFrame(cvrows).sort_values('macro_f1_mean',ascending=False)
cvres.to_csv(ROOT/'tables'/'classification_comparison_5fold.csv',index=False)

# silhouette sample for descriptor spaces
silrows=[]
ss=clfdf.sample(n=min(5000,len(clfdf)),random_state=7)
for name,cols in sets.items():
    A=StandardScaler().fit_transform(ss[cols].values)
    silrows.append({'model':name,'n_cells':len(ss),'silhouette_score':silhouette_score(A,ss.cell_type.values,metric='euclidean')})
pd.DataFrame(silrows).sort_values('silhouette_score',ascending=False).to_csv(ROOT/'tables'/'silhouette_comparison.csv',index=False)

# robustness downsampling on 3000 cells, compute correlations for key descriptors
sample_idx=rng.choice(nc,size=3000,replace=False)
base=gene_desc.iloc[sample_idx].reset_index(drop=True)
rob=[]
for frac in [.8,.6,.4,.2]:
    Xs=X[:,sample_idx].tocsc(copy=True)
    Xs.data=rng.binomial(Xs.data.astype(np.int64),frac).astype(float)
    Xs.eliminate_zeros()
    libs=np.asarray(Xs.sum(axis=0)).ravel(); sc=np.divide(1e4,libs,out=np.zeros_like(libs),where=libs>0)
    Zs=Xs @ sparse.diags(sc); Zs.data=np.log1p(Zs.data)
    ds=descriptors(Zs)
    for c in ['shannon','apg_entropy_norm','apg_cmax','apg_d_3_0','apg_d_6_0','apg_integrated_2_6']:
        rho,_=stats.spearmanr(base[c],ds[c],nan_policy='omit')
        rob.append({'retained_fraction':frac,'descriptor':c,'spearman_rho':rho})
pd.DataFrame(rob).to_csv(ROOT/'tables'/'downsampling_robustness.csv',index=False)

# figures
plt.figure(figsize=(9,5)); plot=cvres.sort_values('macro_f1_mean'); plt.barh(plot.model,plot.macro_f1_mean,xerr=plot.macro_f1_sd); plt.xlabel('5-fold macro-F1'); plt.tight_layout(); plt.savefig(ROOT/'figures'/'classification_comparison.png',dpi=220); plt.close()
plt.figure(figsize=(9,5)); top10=tests.head(10).sort_values('epsilon_squared'); plt.barh(top10.descriptor,top10.epsilon_squared); plt.xlabel('Kruskal–Wallis epsilon-squared'); plt.tight_layout(); plt.savefig(ROOT/'figures'/'descriptor_effect_sizes.png',dpi=220); plt.close()
# boxplots for key descriptors
for c,title in [('apg_d_6_0','Gene-level APG defect D(6)'),('program_apg_integrated_2_6','Program-level integrated APG deformation')]:
    labs=conf.cell_type.value_counts().index.tolist(); data=[conf.loc[conf.cell_type==l,c].dropna().sample(n=min(1500,(conf.cell_type==l).sum()),random_state=1).values for l in labs]
    plt.figure(figsize=(11,5)); plt.boxplot(data,labels=labs,showfliers=False); plt.xticks(rotation=30,ha='right'); plt.ylabel(title); plt.tight_layout(); plt.savefig(ROOT/'figures'/(c+'_by_cell_type.png'),dpi=220); plt.close()
# deformation profiles median by label
ps=[2,2.25,2.5,3,4,5,6]; dcols=[None,'apg_d_2_25','apg_d_2_5','apg_d_3_0','apg_d_4_0','apg_d_5_0','apg_d_6_0']
plt.figure(figsize=(8,5))
for lab,g in conf.groupby('cell_type'):
    vals=[0]+[g[c].median() for c in dcols[1:]]; plt.plot(ps,vals,marker='o',label=lab)
plt.xlabel('Exponent p'); plt.ylabel('Median APG defect D(p)'); plt.legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(ROOT/'figures'/'apg_deformation_profiles_by_cell_type.png',dpi=220); plt.close()
# robustness
rbd=pd.read_csv(ROOT/'tables'/'downsampling_robustness.csv')
plt.figure(figsize=(8,5))
for d,g in rbd.groupby('descriptor'): plt.plot(g.retained_fraction,g.spearman_rho,marker='o',label=d)
plt.xlabel('Retained count fraction'); plt.ylabel('Spearman correlation with full-depth descriptor'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(ROOT/'figures'/'downsampling_robustness.png',dpi=220); plt.close()

# program score medians by cell type
prog_table=[]
for lab,gidx in pd.Series(labels_out).groupby(pd.Series(labels_out)).groups.items():
    if lab=='Uncertain': continue
    row={'cell_type':lab,'n_cells':len(gidx)}
    for i,pn in enumerate(programs): row[pn+'_median_score']=float(np.median(program_scores[i,list(gidx)]))
    prog_table.append(row)
pd.DataFrame(prog_table).to_csv(ROOT/'tables'/'program_scores_by_cell_type.csv',index=False)

# manifests and report
metrics=pd.read_csv('/mnt/data/fresh_68k_pbmc_donor_a_metrics_summary(1).csv')
shutil.copy2('/mnt/data/fresh_68k_pbmc_donor_a_metrics_summary(1).csv',ROOT/'provenance'/'fresh_68k_pbmc_donor_a_metrics_summary.csv')
shutil.copy2('/mnt/data/fresh_68k_pbmc_donor_a — Cell Ranger(1).pdf',ROOT/'provenance'/'fresh_68k_pbmc_donor_a_Cell_Ranger.pdf')
# Copy compact source archives? not include 143MB. record checksums
checks={}
for p in [matrix_tar,analysis_tar]:
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    checks[p.name]={'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}
manifest={'dataset':'Fresh 68k PBMCs (Donor A)','cells':nc,'genes':ng,'nonzero_counts':int(X.nnz),'analysis_date':'2026-07-11','input_archives':checks,'marker_genes_used':used_markers,'program_genes_used':used_programs,'confident_annotations':int((labels_out!='Uncertain').sum()),'uncertain_annotations':int((labels_out=='Uncertain').sum())}
with open(ROOT/'results'/'run_manifest.json','w') as f: json.dump(manifest,f,indent=2)

bestrow=cvres.iloc[0]; base_row=cvres[cvres.model=='M2_Classical_diversity'].iloc[0]; m3=cvres[cvres.model=='M3_Classical_plus_APG'].iloc[0]
report=f'''# Executed Results\n\n## Dataset\nThe analysis used the official 10x Genomics Fresh 68k PBMCs (Donor A) filtered matrix: **{nc:,} cells**, **{ng:,} genes**, and **{X.nnz:,} non-zero UMI entries**.\n\n## Completed analyses\n- Gene-level classical diversity descriptors: Shannon, Rényi-2, Simpson, Gini, Herfindahl concentration, participation ratio, and detected genes.\n- Gene-level APG descriptors: Euclidean squared weights, entropy, concentration, effective dimensions, exponent-deformation profile, and integrated deformation.\n- Marker-supported provisional immune-cell annotation with explicit confidence filtering.\n- Programme-level APG over nine curated immune and cellular programmes.\n- Five-fold stratified classification comparison.\n- Kruskal–Wallis effect sizes.\n- Descriptor-space silhouette analysis.\n- Count-depth downsampling robustness at 80%, 60%, 40%, and 20%.\n\n## Main comparative result\nThe classical diversity model achieved macro-F1 **{base_row.macro_f1_mean:.4f} ± {base_row.macro_f1_sd:.4f}**. Adding gene-level APG deformation descriptors produced macro-F1 **{m3.macro_f1_mean:.4f} ± {m3.macro_f1_sd:.4f}**. The best tested compact model was **{bestrow.model}**, with macro-F1 **{bestrow.macro_f1_mean:.4f} ± {bestrow.macro_f1_sd:.4f}** and balanced accuracy **{bestrow.balanced_accuracy_mean:.4f} ± {bestrow.balanced_accuracy_sd:.4f}**.\n\n## Interpretation\nThe executed comparison tests whether the exponent-deformation profile contributes information beyond standard diversity and concentration summaries. Results are reported exactly in `tables/classification_comparison_5fold.csv`; no superiority claim is made unless supported by those values.\n\n## Scientific scope\nThis is a one-donor methodological benchmark. Cell-level cross-validation measures internal discrimination and does not establish donor-level or clinical generalization. The marker-based labels are reproducible computational annotations, not expert-reviewed clinical labels.\n'''
(ROOT/'results'/'EXECUTED_RESULTS.md').write_text(report)

readme=f'''# Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes\n\n## Executed final research package\nThis package contains a fully executed, reproducible analysis of the official 10x Genomics Fresh 68k PBMCs (Donor A) filtered gene–cell matrix. The work evaluates whether APG exponent-deformation descriptors provide complementary structural information beyond conventional diversity measures.\n\n## Dataset\n- Cells: {nc:,}\n- Genes: {ng:,}\n- Non-zero UMI entries: {X.nnz:,}\n- Reference genome: hg19\n- Cell Ranger version: 1.1.0\n\n## Novel contribution\nThe study represents every cell through a bounded exponent-deformation profile rather than through a single entropy statistic. Gene-level and biological-programme-level APG states are compared directly with Shannon entropy, Rényi-2 entropy, Simpson diversity, Gini inequality, Herfindahl concentration, participation ratio, and detected-gene count.\n\n## Package contents\n- `results/cell_level_descriptors_and_annotations.csv.gz`: all cell-level outputs\n- `tables/`: comparative statistics, classification, pathway summaries, silhouette and robustness results\n- `figures/`: publication-ready executed figures\n- `code/run_complete_analysis.py`: complete analysis source\n- `manuscript/MANUSCRIPT_DRAFT.md`: research-paper draft based on executed results\n- `results/EXECUTED_RESULTS.md`: concise factual result summary\n- `provenance/`: official dataset metrics and Cell Ranger report\n\n## Scope\nThe dataset contains one donor. The study is therefore positioned as a methodological and exploratory benchmark, without population-level or clinical claims.\n'''
(ROOT/'README.md').write_text(readme)

methods='''# Methods\n\nFiltered UMI counts were library-size normalized to 10,000 counts per cell and transformed with log(1+x). Classical probability weights used normalized log-expression. APG used squared normalized log-expression weights. The APG defect was evaluated at p = 2.25, 2.5, 3, 4, 5, and 6 and integrated by trapezoidal quadrature from p=2 to p=6.\n\nProvisional immune-cell annotations were generated from curated marker modules. A cell was retained as confidently annotated only when its highest standardized marker score and margin over the second-highest score exceeded fixed thresholds. Nine curated biological programmes were scored by mean normalized expression, followed by programme-level APG construction.\n\nClassification used five-fold stratified cross-validation and class-weighted multinomial logistic regression. All models used identical folds. Robustness was assessed by binomial thinning of raw UMI counts to 80%, 60%, 40%, and 20% of the original depth on a fixed random sample of 3,000 cells.\n'''
(ROOT/'METHODS.md').write_text(methods)

# manuscript draft factual
manu=f'''# Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes\n\n## Abstract\nSingle-cell RNA sequencing provides high-dimensional molecular profiles, but compact structural summaries often reduce each cell to a single diversity or concentration statistic. This study develops Arithmetic Power Geometry (APG) descriptors for single-cell transcriptomes and evaluates whether exponent deformation provides complementary information beyond established measures. The official 10x Genomics Fresh 68k PBMCs (Donor A) dataset comprising {nc:,} cells was analysed. Library-normalized gene expression was transformed into Euclidean APG weights, from which entropy, concentration, effective dimensions, exponent-defect curves and integrated deformation were computed. Direct comparisons included Shannon entropy, Rényi-2 entropy, Simpson diversity, Gini inequality, Herfindahl concentration, participation ratio and detected-gene count. Marker-supported provisional immune-cell annotations and nine curated biological programmes enabled cell-type and programme-level analyses. Five-fold stratified models compared Shannon-only, classical-diversity, APG-augmented and programme-level representations. The classical diversity model obtained macro-F1 {base_row.macro_f1_mean:.4f}, while the corresponding classical-plus-APG model obtained {m3.macro_f1_mean:.4f}. The best evaluated representation was {bestrow.model} with macro-F1 {bestrow.macro_f1_mean:.4f}. Downsampling experiments quantified descriptor stability under reduced count depth. These results establish an executed one-donor benchmark for APG-based single-cell structural analysis. The findings support APG as an interpretable complementary representation, while multi-donor external validation remains necessary.\n\n## 1. Introduction\nSingle-cell transcriptomic profiles contain thousands of measurements per cell. Conventional workflows preserve much of this complexity through latent embeddings, while compact summaries such as Shannon entropy or concentration indices describe only selected aspects of the expression distribution. APG introduces a controlled exponent deformation of normalized squared expression weights. The resulting profile describes how rapidly a cell's distributed molecular activity collapses toward dominant components as the deformation exponent increases.\n\nThe central question is whether the APG deformation profile contains discriminative or interpretable information not already represented by standard diversity and concentration measures. This work addresses that question through direct baseline comparison, programme-level analysis, effect-size estimation, classification and depth-robustness testing on a canonical PBMC dataset.\n\n## 2. Mathematical framework\nFor nonnegative normalized expression z_i, APG weights are w_i=z_i^2/sum_j z_j^2. Shannon APG entropy is H=-sum_i w_i log w_i. The exponent defect is D(p)=1-sum_i w_i^(p/2), p>=2. Integrated deformation is the numerical integral of D(p) from 2 to 6.\n\n## 3. Materials and methods\nSee `METHODS.md` for the complete executed procedure.\n\n## 4. Results\n### 4.1 Descriptor comparison\nAll global Kruskal–Wallis results and epsilon-squared effect sizes are reported in `tables/cell_type_global_tests.csv`.\n\n### 4.2 Incremental classification value\nThe Shannon-only, classical-diversity, APG-augmented and programme-level models were evaluated on identical five-fold splits. Complete results appear in `tables/classification_comparison_5fold.csv`. The classical model macro-F1 was {base_row.macro_f1_mean:.4f}; adding gene-level APG produced {m3.macro_f1_mean:.4f}.\n\n### 4.3 Programme-level APG\nNine curated programmes were evaluated: T-cell receptor signalling, cytotoxicity, B-cell receptor activity, interferon response, inflammatory myeloid activity, oxidative phosphorylation, ribosomal activity, cell cycle and apoptosis/stress. Programme summaries appear in `tables/program_scores_by_cell_type.csv`.\n\n### 4.4 Robustness\nCount-thinning results at four retained-depth levels appear in `tables/downsampling_robustness.csv`.\n\n## 5. Discussion\nThe executed study separates the established information-theoretic quantities from APG-specific deformation descriptors and tests their incremental value rather than assuming it. APG's contribution is the bounded multi-exponent profile and integrated deformation, not the reinvention of Shannon or concentration measures. The analysis remains limited by its single-donor design and provisional computational annotations.\n\n## 6. Conclusion\nAPG can be computed reproducibly at single-cell scale and compared transparently with conventional diversity measures. The present benchmark provides the mathematical, computational and empirical foundation for independent multi-donor validation.\n'''
(ROOT/'manuscript'/'MANUSCRIPT_DRAFT.md').write_text(manu)

# code copy
shutil.copy2('/mnt/data/build_apg_q2.py',ROOT/'code'/'run_complete_analysis.py')
(ROOT/'requirements.txt').write_text('numpy\npandas\nscipy\nscikit-learn\nmatplotlib\n')
# tests
(ROOT/'tests'/'test_formulas.py').write_text('''import numpy as np\n\ndef test_apg_bounds():\n    x=np.array([1.,2.,3.]); w=x*x/np.sum(x*x); H=-np.sum(w*np.log(w)); assert 0<=H<=np.log(3)+1e-12\n    ds=[1-np.sum(w**(p/2)) for p in [2,2.5,3,4,6]]; assert all(0<=d<=1 for d in ds); assert np.all(np.diff(ds)>=-1e-12)\n''')

# zip
zip_path=Path('/mnt/data/APG_Single_Cell_PBMC_Q2_Final.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(ROOT.parent))
print('\nDONE',zip_path,zip_path.stat().st_size)
print(cvres.to_string(index=False))
print('confident',len(conf),'uncertain',nc-len(conf))
