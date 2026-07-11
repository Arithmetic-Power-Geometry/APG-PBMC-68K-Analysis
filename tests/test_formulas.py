import numpy as np
def test_apg_bounds():
 x=np.array([1.,2.,3.]); w=x*x/np.sum(x*x); ds=[1-np.sum(w**(p/2)) for p in [2,2.5,3,4,6]]; assert all(0<=d<=1 for d in ds); assert np.all(np.diff(ds)>=-1e-12)
