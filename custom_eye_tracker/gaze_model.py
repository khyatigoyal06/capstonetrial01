"""Ridge regression baseline with transparent target-location model selection."""
import json
from pathlib import Path
import numpy as np

def _design(x):
    x=np.asarray(x,float)
    # Median-impute missing eye geometry, then standardize using training only.
    med=np.nanmedian(x,axis=0); med=np.where(np.isfinite(med),med,0.0)
    return np.where(np.isfinite(x),x,med)

class RidgeGazeModel:
    def __init__(self, alpha=10.0): self.alpha=float(alpha); self.fitted=False
    def fit(self, features, targets):
        x=np.asarray(features,float); y=np.asarray(targets,float); self.median=np.nanmedian(x,axis=0)
        self.median=np.where(np.isfinite(self.median),self.median,0.0); x=np.where(np.isfinite(x),x,self.median)
        self.mean=x.mean(axis=0); self.scale=x.std(axis=0); self.scale=np.where(self.scale<1e-8,1.0,self.scale)
        z=(x-self.mean)/self.scale; z=np.c_[np.ones(len(z)),z]
        penalty=np.eye(z.shape[1])*self.alpha; penalty[0,0]=0
        self.coef_=np.linalg.solve(z.T@z+penalty,z.T@y); self.fitted=True; return self
    def predict(self, features):
        x=np.asarray(features,float); single=x.ndim==1; x=np.atleast_2d(x); x=np.where(np.isfinite(x),x,self.median)
        result=np.c_[np.ones(len(x)),(x-self.mean)/self.scale]@self.coef_
        return result[0] if single else result
    def to_dict(self):
        return {"kind":"ridge","alpha":self.alpha,"median":self.median.tolist(),"mean":self.mean.tolist(),"scale":self.scale.tolist(),"coef":self.coef_.tolist()}
    @classmethod
    def from_dict(cls,d):
        o=cls(d["alpha"]); o.median=np.array(d["median"]); o.mean=np.array(d["mean"]); o.scale=np.array(d["scale"]); o.coef_=np.array(d["coef"]); o.fitted=True; return o
    def save(self,path,metadata): Path(path).write_text(json.dumps({"version":1,"metadata":metadata,"model":self.to_dict()},indent=2))
    @classmethod
    def load(cls,path,expected_metadata=None):
        d=json.loads(Path(path).read_text())
        if d.get("version")!=1: raise ValueError("Unsupported model version")
        if expected_metadata and d["metadata"]!=expected_metadata: raise ValueError("Model metadata mismatch; recalibrate for this setup")
        return cls.from_dict(d["model"]),d["metadata"]

def select_alpha(train_x,train_y,val_x,val_y,candidates=(0.1,1,10,100)):
    scores=[]
    for a in candidates:
        m=RidgeGazeModel(a).fit(train_x,train_y)
        scores.append((float(np.median(np.linalg.norm(m.predict(val_x)-val_y,axis=1))),float(a)))
    alpha=min(scores)[1]
    return RidgeGazeModel(alpha).fit(train_x,train_y), {"validation_median_by_alpha":{str(a):e for e,a in scores},"selected_alpha":alpha}
