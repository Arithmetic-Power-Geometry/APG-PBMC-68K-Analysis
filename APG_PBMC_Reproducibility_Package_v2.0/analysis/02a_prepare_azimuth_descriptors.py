from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]; WORK=BASE/'work'; OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
X=np.load(WORK/'ADT.npy').T.astype(float)  # cells x features, nonnegative normalized ADT
n=X.shape[1]; eps=1e-15
# classical summaries on ordinary normalized ADT weights
S=X.sum(1,keepdims=True); q=np.divide(X,S,out=np.zeros_like(X),where=S>0); qsafe=np.where(q>0,q,1)
sh=-(q*np.log(qsafe)).sum(1); hhi=(q*q).sum(1); ren=-np.log(np.maximum(hhi,eps)); sim=1-hhi; pr=1/np.maximum(hhi,eps); det=(X>0).sum(1)
# Gini across all features
Xs=np.sort(X,axis=1); idx=np.arange(1,n+1); sums=Xs.sum(1); gini=np.divide(2*(Xs*idx).sum(1),n*sums,out=np.zeros(len(X)),where=sums>0)-(n+1)/n
classical=np.c_[sh,ren,sim,hhi,pr,gini,det].astype('float32'); np.save(OUT/'adt_classical.npy',classical)
# APG summaries
X2=X*X; S2=X2.sum(1,keepdims=True); w=np.divide(X2,S2,out=np.zeros_like(X2),where=S2>0); wsafe=np.where(w>0,w,1)
H=-(w*np.log(wsafe)).sum(1); Hn=H/np.log(n); cmax=w.max(1); c2=(w*w).sum(1); neff=np.exp(H); apr=1/np.maximum(c2,eps)
ps=[2.25,2.5,3.,4.,5.,6.]; ds=[1-np.power(w,p/2).sum(1) for p in ps]; integ=np.trapezoid(np.c_[np.zeros(len(X)),*ds],x=np.array([2,*ps]),axis=1)
apg=np.c_[Hn,cmax,c2,neff,apr,*ds,integ].astype('float32'); np.save(OUT/'adt_apg.npy',apg)
