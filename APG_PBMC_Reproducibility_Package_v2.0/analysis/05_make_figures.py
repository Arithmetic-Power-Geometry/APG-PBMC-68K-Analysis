from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
BASE=Path(__file__).resolve().parents[1]; out=BASE/'manuscript'/'figures'; res=BASE/'results'; out.mkdir(parents=True,exist_ok=True)
# Fig external benchmark
s=pd.read_csv(res/'azimuth_strict_5fold_summary.csv')
order=['RNA PCA (50)','RNA PCA + ADT classical','RNA PCA + ADT APG','RNA PCA + ADT classical + APG','ADT PCA (30)','ADT PCA + classical','ADT PCA + APG','ADT PCA + classical + APG']
s=s.set_index('model').loc[order].reset_index()
fig,ax=plt.subplots(figsize=(8.2,5.2)); y=np.arange(len(s)); ax.barh(y,s.macro_f1_mean,xerr=s.macro_f1_sd,capsize=3); ax.set_yticks(y,s.model); ax.set_xlabel('Macro-F1 (five-fold mean ± SD)'); ax.set_xlim(0.90,0.97); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(out/'external_azimuth_benchmark.pdf',bbox_inches='tight'); fig.savefig(out/'external_azimuth_benchmark.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Fresh ablation
s=pd.read_csv(res/'fresh68k_ablation_summary.csv')
order=['Programme classical','Programme APG','Programme classical + APG','Gene classical','Gene classical + APG','All classical','All classical + all APG']
s=s.set_index('model').loc[order].reset_index()
fig,ax=plt.subplots(figsize=(8.2,4.8)); y=np.arange(len(s)); ax.barh(y,s.macro_f1_mean,xerr=s.macro_f1_sd,capsize=3); ax.set_yticks(y,s.model); ax.set_xlabel('Macro-F1 (five-fold mean ± SD)'); ax.set_xlim(0.20,0.58); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(out/'programme_ablation.pdf',bbox_inches='tight'); fig.savefig(out/'programme_ablation.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Entropy relationship vs exponent
r=pd.read_csv(res/'fresh68k_entropy_redundancy.csv'); mapx={'apg_d_2_25':2.25,'apg_d_2_5':2.5,'apg_d_3_0':3,'apg_d_4_0':4,'apg_d_5_0':5,'apg_d_6_0':6}; q=r[r.descriptor.isin(mapx)].copy(); q['p']=q.descriptor.map(mapx); q=q.sort_values('p')
fig,ax=plt.subplots(figsize=(6.8,4.2)); ax.plot(q.p,q.linear_r2_from_apg_entropy_norm,marker='o'); ax.set_xlabel('Exponent p'); ax.set_ylabel(r'$R^2$ from normalized APG entropy'); ax.set_ylim(.75,1.01); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'entropy_redundancy_by_exponent.pdf',bbox_inches='tight'); fig.savefig(out/'entropy_redundancy_by_exponent.png',dpi=300,bbox_inches='tight'); plt.close(fig)
