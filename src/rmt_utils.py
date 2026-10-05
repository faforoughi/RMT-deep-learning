import numpy as np

def gram_eigenvalues(weight):
    W=np.asarray(weight,dtype=np.float64)
    if W.ndim!=2: raise ValueError("weight must be 2-D")
    n,p=W.shape
    X=W-W.mean(axis=1,keepdims=True)
    s=X.std()
    X=X/(s if s>1e-12 else 1.0)
    G=(X@X.T)/p if n<=p else (X.T@X)/n
    return np.clip(np.linalg.eigvalsh(G),0,None)

def mp_support(gamma):
    if not 0<gamma<=1: raise ValueError("gamma must be in (0,1]")
    s=np.sqrt(gamma); return (1-s)**2,(1+s)**2

def mp_density(x,gamma):
    x=np.asarray(x,dtype=float); lo,hi=mp_support(gamma)
    y=np.zeros_like(x); m=(x>lo)&(x<hi)&(x>0)
    y[m]=np.sqrt((hi-x[m])*(x[m]-lo))/(2*np.pi*gamma*x[m])
    return y

def matrix_aspect_ratio(W):
    n,p=np.asarray(W).shape; return min(n,p)/max(n,p)

def spectral_summary(W):
    v=gram_eigenvalues(W); g=matrix_aspect_ratio(W); lo,hi=mp_support(g)
    q=v/(v.sum()+1e-12)
    return {"gamma":float(g),"mp_lower":float(lo),"mp_upper":float(hi),
    "lambda_max":float(v.max()),"lambda_mean":float(v.mean()),
    "fraction_above_mp":float(np.mean(v>hi)),
    "effective_rank":float(np.exp(-np.sum(q*np.log(q+1e-12))))}
