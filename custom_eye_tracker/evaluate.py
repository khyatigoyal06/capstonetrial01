import numpy as np
from .smoothing import ExponentialSmoother

def summarize(targets,predictions,valid,thresholds=(50,100,150)):
    t=np.asarray(targets,float); p=np.asarray(predictions,float); v=np.asarray(valid,bool); errs=np.linalg.norm(p-t,axis=1); good=errs[v & np.isfinite(errs)]
    regions={}
    if len(t):
        mx,my=np.nanmax(t[:,0]),np.nanmax(t[:,1])
        for name,mask in {"top_left":(t[:,0]<mx/2)&(t[:,1]<my/2),"top_right":(t[:,0]>=mx/2)&(t[:,1]<my/2),"bottom_left":(t[:,0]<mx/2)&(t[:,1]>=my/2),"bottom_right":(t[:,0]>=mx/2)&(t[:,1]>=my/2)}.items():
            e=errs[mask&v&np.isfinite(errs)]; regions[name]={"n_locations":int(mask.sum()),"median_px":float(np.median(e)) if len(e) else None}
    return {"locations":int(len(t)),"valid_locations":int(len(good)),"mean_px":float(np.mean(good)) if len(good) else None,"median_px":float(np.median(good)) if len(good) else None,"p90_px":float(np.percentile(good,90)) if len(good) else None,"worst_px":float(np.max(good)) if len(good) else None,"within_px":{str(a):float(np.mean(good<=a)) if len(good) else None for a in thresholds},"coverage":float(np.mean(v)) if len(v) else 0.0,"regions":regions}

def select_smoothing_tau(model, target_rows, candidates, thresholds=(50,100,150)):
    """Choose smoothing by validation-target median error; frames stay grouped by location."""
    targets=np.asarray([r["target_px"] for r in target_rows],float)
    settings={}
    for tau in candidates:
        points=[]
        for row in target_rows:
            samples=row["raw_samples"]
            preds=model.predict([s["features"] for s in samples])
            smoother=ExponentialSmoother(tau); values=[]
            for sample,pred in zip(samples,preds):
                values.append(smoother.update(pred,sample["timestamp_monotonic_s"]))
            points.append(np.median(values,axis=0))
        settings[float(tau)]=summarize(targets,points,[True]*len(points),thresholds)
    selected=min(settings,key=lambda tau:(settings[tau]["median_px"],tau))
    return selected,settings
