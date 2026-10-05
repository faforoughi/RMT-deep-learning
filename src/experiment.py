import argparse,json
from pathlib import Path
import numpy as np, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.model import MLP,weight_matrices
from src.rmt_utils import spectral_summary

def data(seed=42,batch=128):
    d=load_digits(); X=StandardScaler().fit_transform(d.data).astype("float32"); y=d.target.astype("int64")
    a,b,c,e=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
    return (DataLoader(TensorDataset(torch.tensor(a),torch.tensor(c)),batch_size=batch,shuffle=True),
            DataLoader(TensorDataset(torch.tensor(b),torch.tensor(e)),batch_size=512))

@torch.no_grad()
def acc(model,loader,device):
    model.eval(); ok=n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); ok+=int((model(x).argmax(1)==y).sum()); n+=y.numel()
    return ok/n

def snap(model,epoch,width,a):
    out=[]
    for layer,W in weight_matrices(model).items():
        r=spectral_summary(W); r.update(epoch=epoch,width=width,layer=layer,test_accuracy=a); out.append(r)
    return out

def run(width=256,epochs=20,lr=1e-3,seed=42,out="results"):
    np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr,te=data(seed); model=MLP(width=width).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr); loss=nn.CrossEntropyLoss()
    rows=snap(model,0,width,acc(model,te,device)); checkpoints={1,5,10,epochs}
    for ep in range(1,epochs+1):
        model.train()
        for x,y in tr:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss(model(x),y).backward(); opt.step()
        if ep in checkpoints:
            a=acc(model,te,device); rows+=snap(model,ep,width,a); print(width,ep,round(a,4))
    Path(out).mkdir(exist_ok=True)
    p=Path(out)/f"spectral_width_{width}.json"; p.write_text(json.dumps(rows,indent=2)); return p

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--width",type=int,default=256); p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--seed",type=int,default=42); p.add_argument("--out",default="results")
    print(run(**vars(p.parse_args())))
