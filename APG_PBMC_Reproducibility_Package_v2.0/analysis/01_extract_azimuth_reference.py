import sys, gzip, numpy as np, pandas as pd, pickle, os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
WORK=BASE/'work'; WORK.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rds_parser3 import Parser, deref, attrs_dict, char, Obj

path=str(BASE/'source_data'/'Azimuth_Human_PBMC_ref.Rds')
raw=gzip.decompress(open(path,'rb').read())
p=Parser(raw[2:]); ver=(p.i(),p.i(),p.i()); encn=p.i(); p.f.read(encn); root=p.parse(); rd=attrs_dict(root)

def strvec(o):
    o=deref(o)
    if not o or o.typ!=16:return None
    return [char(x) for x in o.value]
def names(o):
    a=attrs_dict(o); return strvec(a.get('names')) if 'names' in a else None

def as_factor(o):
    o=deref(o)
    # regular integer factor
    candidates=[]
    def walk(x,depth=0):
        x=deref(x)
        if not isinstance(x,Obj) or depth>6: return
        if x.typ==13 and len(x.value)>1000:
            a=attrs_dict(x); lev=strvec(a.get('levels')) if 'levels' in a else None
            candidates.append((x,lev))
        if x.typ==238:
            for y in x.value: walk(y,depth+1)
        elif x.typ in (2,6):
            car,cdr=x.value; walk(car,depth+1); walk(cdr,depth+1)
        elif x.typ==19:
            for y in x.value: walk(y,depth+1)
    walk(o)
    for x,lev in candidates:
        if lev:
            codes=np.array(x.value,dtype=int)
            labs=np.array([lev[c-1] if c>0 and c<=len(lev) else None for c in codes],dtype=object)
            return labs,lev,codes
    raise RuntimeError('factor not found')

def list_named(o):
    o=deref(o); ns=names(o); return {n:deref(v) for n,v in zip(ns,o.value)}

def s4_slots(o): return attrs_dict(o)

def get_matrix_s4(o):
    o=deref(o); sl=attrs_dict(o)
    dim=np.array(deref(sl['Dim']).value,dtype=int)
    x=np.array(deref(sl['x']).value,dtype=float)
    i=np.array(deref(sl['i']).value,dtype=int)
    pp=np.array(deref(sl['p']).value,dtype=int)
    return dim,x,i,pp,sl

def dense_with_dims(o):
    o=deref(o); a=attrs_dict(o); dim=np.array(deref(a['dim']).value,dtype=int) if 'dim' in a else None
    arr=np.array(o.value,dtype=float)
    if dim is not None: arr=arr.reshape(tuple(dim),order='F')
    return arr,a

meta=list_named(rd['meta.data'])
for k in ['celltype.l1','celltype.l2','celltype.l3']:
    labs,lev,codes=as_factor(meta[k]); print(k,len(labs),len(lev)); print(pd.Series(labs).value_counts().head(30).to_string()); np.save(str(WORK/f'{k}.npy'),labs,allow_pickle=True)
# active ident
act=deref(rd['active.ident']); lev=strvec(attrs_dict(act)['levels']); codes=np.array(act.value,dtype=int); labs=np.array([lev[c-1] for c in codes],dtype=object)
print('active ident',len(labs),len(lev)); print(pd.Series(labs).value_counts().head(30).to_string()); np.save(str(WORK/'active_ident.npy'),labs,allow_pickle=True)
cellnames=strvec(attrs_dict(act)['names']); np.save(str(WORK/'cellnames.npy'),np.array(cellnames,dtype=object),allow_pickle=True)
print('prefixes',pd.Series([x.split('_',1)[0] for x in cellnames]).value_counts().to_string())
# reductions
reds=list_named(rd['reductions'])
for rname in ['refDR','refUMAP']:
    slots=attrs_dict(reds[rname]); emb,a=dense_with_dims(slots['cell.embeddings']); print(rname,'emb',emb.shape,emb.min(),emb.max()); np.save(str(WORK/f'{rname}_emb.npy'),emb)
    if 'feature.loadings' in slots:
        fl,fa=dense_with_dims(slots['feature.loadings']); print(rname,'load',fl.shape); np.save(str(WORK/f'{rname}_load.npy'),fl)
# ADT data
assays=list_named(rd['assays']); adt=attrs_dict(assays['ADT']);
adata,a=dense_with_dims(adt['data']); print('ADT',adata.shape,'minmax',np.nanmin(adata),np.nanmax(adata),'mean',np.nanmean(adata)); np.save(str(WORK/'ADT.npy'),adata)
# dimnames of ADT data
dn=deref(attrs_dict(adt['data']).get('dimnames'))
if isinstance(dn,Obj) and dn.typ==19:
    rows=strvec(dn.value[0]); cols=strvec(dn.value[1]); print('ADT row sample',rows[:30]); print('ADT col same',cols[:5]);
    np.save(str(WORK/'ADT_features.npy'),np.array(rows,dtype=object),allow_pickle=True)
