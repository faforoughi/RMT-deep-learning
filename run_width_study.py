from src.experiment import run
if __name__=="__main__":
    for width in [64,128,256,512]: run(width=width,epochs=20,seed=42)
