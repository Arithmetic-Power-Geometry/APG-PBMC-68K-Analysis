import gzip, struct, io, numpy as np
from dataclasses import dataclass
T={0:'NIL',1:'SYM',2:'LIST',3:'CLO',4:'ENV',5:'PROM',6:'LANG',7:'SPECIAL',8:'BUILTIN',9:'CHAR',10:'LGL',13:'INT',14:'REAL',15:'CPLX',16:'STR',17:'DOT',18:'ANY',19:'VEC',20:'EXPR',21:'BCODE',22:'EXTPTR',23:'WEAKREF',24:'RAW',25:'S4',238:'ALTREP',239:'ATTRLIST',240:'ATTRLANG',241:'BASEENV',242:'EMPTYENV',243:'BCREPREF',244:'BCREPDEF',245:'GENERICREF',246:'CLASSREF',247:'PERSIST',248:'PACKAGE',249:'NAMESPACE',250:'BASENAMESPACE',251:'MISSINGARG',252:'UNBOUND',253:'GLOBALENV',254:'NILVALUE',255:'REF'}
@dataclass
class Obj:
    typ:int; value:object=None; attrs:object=None; tag:object=None; gp:int=0; obj:bool=False; ref:int=0; referenced:object=None
    @property
    def t(self): return T.get(self.typ,str(self.typ))
class Parser:
    def __init__(self,data): self.f=io.BytesIO(data); self.refs=[]
    def i(self):
        b=self.f.read(4)
        if len(b)!=4: raise EOFError(self.f.tell())
        return struct.unpack('>i',b)[0]
    def u(self): return struct.unpack('>I',self.f.read(4))[0]
    def length(self):
        n=self.i()
        if n==-1:
            hi=self.u(); lo=self.u(); return (hi<<32)+lo
        if n<0: raise ValueError(('neg length',n,self.f.tell()))
        return n
    def d_arr(self,n): return np.frombuffer(self.f.read(8*n),dtype='>f8').astype('f8')
    def i_arr(self,n): return np.frombuffer(self.f.read(4*n),dtype='>i4').astype('i4')
    def c_arr(self,n): return np.frombuffer(self.f.read(16*n),dtype='>c16').astype('c16')
    def parse_info(self,x):
        typ=x&0xff
        if typ==255:
            idx=(x>>8)&0xffffff
            return typ,False,False,False,0,idx
        return typ,bool((x>>8)&1),bool((x>>9)&1),bool((x>>10)&1),(x>>12)&0xffff,0
    def read_stringvec_special(self):
        names=self.i()
        if names!=0: raise ValueError(('special names',names))
        n=self.length(); return [self.parse() for _ in range(n)]
    def parse_bc_lang(self, typ, reps):
        if typ==243:
            pos=self.i(); return reps[pos]
        if typ in (244,6,2,240,239):
            pos=-1; has_attr=False
            if typ==244:
                pos=self.i(); typ=self.i()
            if typ==240: typ=6; has_attr=True
            elif typ==239: typ=2; has_attr=True
            ans=Obj(typ)
            if pos>=0: reps[pos]=ans
            attrs=self.parse() if has_attr else None
            tag=self.parse()
            car=self.parse_bc_lang(self.i(),reps)
            cdr=self.parse_bc_lang(self.i(),reps)
            ans.attrs=attrs; ans.tag=tag; ans.value=(car,cdr)
            return ans
        # pad/type was already read; default ignores it and reads a normal item
        return self.parse()
    def parse_bc1(self,reps):
        code=self.parse()
        n=self.i(); const=[]
        for _ in range(n):
            typ=self.i()
            if typ==21: c=self.parse_bc1(reps)
            elif typ in (6,2,244,243,240,239): c=self.parse_bc_lang(typ,reps)
            else: c=self.parse()  # typ is an informational TYPEOF marker, serialized item follows
            const.append(c)
        return Obj(21,value=(code,const))
    def parse(self, info_int=None):
        x=self.i() if info_int is None else info_int
        typ,of,af,tf,gp,ref=self.parse_info(x)
        if typ==255:
            if ref==0: ref=self.i()
            if ref<1 or ref>len(self.refs): raise ValueError(f'bad ref {ref} at {self.f.tell()} refs {len(self.refs)}')
            return Obj(typ,ref=ref,referenced=self.refs[ref-1])
        if typ in (254,241,242,250,251,252,253,0): return Obj(typ,gp=gp,obj=of)
        # reference-like immediate cases
        if typ==1:
            o=Obj(typ); self.refs.append(o); o.value=self.parse(); return o
        if typ in (248,249,247):
            vals=self.read_stringvec_special(); o=Obj(typ,value=vals); self.refs.append(o); return o
        if typ==4:
            o=Obj(typ,obj=True); self.refs.append(o)
            locked=bool(self.i()); enc=self.parse(); frame=self.parse(); ht=self.parse(); attrs=self.parse()
            o.value=(locked,enc,frame,ht); o.attrs=attrs; return o
        if typ==22:
            o=Obj(typ); self.refs.append(o); o.value=(self.parse(),self.parse())
            if af: o.attrs=self.parse()
            return o
        if typ==23:
            # weakref saved without payload in older serialization handling; attributes may follow
            o=Obj(typ); self.refs.append(o)
            if af: o.attrs=self.parse()
            return o
        tag=attrs=None; value=None
        if typ in (2,6,3,5,17):
            if af: attrs=self.parse()
            if tf: tag=self.parse()
            car=self.parse(); cdr=self.parse(); value=(car,cdr)
            return Obj(typ,value,attrs,tag,gp,of)
        if typ in (7,8):
            n=self.i(); value=self.f.read(n)
        elif typ==9:
            n=self.i(); value=None if n==-1 else self.f.read(n)
        elif typ in (10,13):
            n=self.length(); value=self.i_arr(n)
        elif typ==14:
            n=self.length(); value=self.d_arr(n)
        elif typ==15:
            n=self.length(); value=self.c_arr(n)
        elif typ in (16,19,20):
            n=self.length(); value=[self.parse() for _ in range(n)]
        elif typ==24:
            n=self.length(); value=self.f.read(n)
        elif typ==25:
            value=None
        elif typ==238:
            # ALTREP: class info, state, attrs are each normal serialized items; no extra attrs pass below
            value=(self.parse(),self.parse(),self.parse())
            return Obj(typ,value=value,gp=gp,obj=of)
        elif typ==21:
            nrep=self.i(); reps=[None]*nrep
            return self.parse_bc1(reps)
        elif typ in (243,244,239,240):
            raise ValueError(('BC marker outside BC',typ,self.f.tell()))
        elif typ in (245,246): raise NotImplementedError(('class/generic ref',typ,self.f.tell()))
        else: raise NotImplementedError((typ,T.get(typ),self.f.tell(),hex(x)))
        o=Obj(typ,value,attrs,tag,gp,of)
        if af: o.attrs=self.parse()
        return o

def deref(o):
    while isinstance(o,Obj) and o.typ==255: o=o.referenced
    return o
def char(o):
    o=deref(o)
    if o and o.typ==9 and isinstance(o.value,(bytes,bytearray)): return o.value.decode('utf-8','replace')
    if o and o.typ==1: return char(o.value)
    return None
def pairlist_to_dict(o,maxn=100000):
    d={}; n=0; o=deref(o)
    while isinstance(o,Obj) and o.typ in (2,6):
        car,cdr=o.value; key=char(o.tag) if o.tag else None
        d[key if key is not None else f'_{n}']=deref(car)
        o=deref(cdr); n+=1
        if n>maxn: raise RuntimeError('pairlist loop')
    return d
def attrs_dict(o):
    o=deref(o); return pairlist_to_dict(o.attrs) if isinstance(o,Obj) and o.attrs else {}
def obj_summary(o):
    o=deref(o)
    if not isinstance(o,Obj): return repr(o)
    s=o.t
    if o.typ==9: s+=f'({char(o)})'
    elif o.typ in (10,13,14,15,16,19,20): s+=f'[{len(o.value)}]'
    if o.attrs:
        try:s+=f' attrs={list(pairlist_to_dict(o.attrs).keys())[:15]}'
        except Exception as e:s+=f' attrs=?{e}'
    return s
if __name__=='__main__':
 import sys
 raw=gzip.decompress(open(sys.argv[1],'rb').read()) if open(sys.argv[1],'rb').read(2)==b'\x1f\x8b' else open(sys.argv[1],'rb').read()
 assert raw[:2] in (b'X\n',b'B\n',b'A\n')
 p=Parser(raw[2:]); ver=(p.i(),p.i(),p.i())
 if ver[0]>=3:
    encn=p.i(); enc=p.f.read(encn)
 else: enc=b''
 print('ver',ver,'enc',enc)
 root=p.parse(); print('pos',p.f.tell(),'len',len(raw)-2,'refs',len(p.refs),'root',obj_summary(root))
 print('root attrs',list(attrs_dict(root).keys()))
 for k,v in attrs_dict(root).items(): print('SLOT',k,obj_summary(v))
