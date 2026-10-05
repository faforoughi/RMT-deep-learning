import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt

def main(paths,output):
    rows=[]
    for p in paths: rows+=json.loads(Path(p).read_text())
    layers={r["layer"] for r in rows}; target="net.2" if "net.2" in layers else sorted(layers)[0]
    d=[r for r in rows if r["layer"]==target]
    plt.figure(figsize=(8,5))
    for w in sorted({r["width"] for r in d}):
        z=sorted([r for r in d if r["width"]==w],key=lambda r:r["epoch"])
        plt.plot([r["epoch"] for r in z],[r["lambda_max"]/r["mp_upper"] for r in z],marker="o",label=f"width={w}")
    plt.axhline(1,ls="--",lw=1,label="MP upper edge"); plt.xlabel("Epoch"); plt.ylabel("lambda_max / lambda_MP+")
    plt.title("Spectral departure from the Marchenko-Pastur upper edge"); plt.legend(); plt.tight_layout()
    Path(output).parent.mkdir(parents=True,exist_ok=True); plt.savefig(output,dpi=180); print(output)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("paths",nargs="+"); p.add_argument("--output",default="results/figures/spectral_dynamics.png")
    a=p.parse_args(); main(a.paths,a.output)
