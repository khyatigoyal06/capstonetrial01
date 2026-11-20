import numpy as np

class ExponentialSmoother:
    def __init__(self,tau_s=0.12): self.tau=float(tau_s); self.value=None; self.time=None
    def update(self,value,timestamp):
        p=np.asarray(value,float)
        if not np.isfinite(p).all(): return None
        if self.value is None or timestamp<=self.time: self.value=p.copy()
        else:
            dt=timestamp-self.time; w=1-np.exp(-dt/max(self.tau,1e-6)); self.value=self.value+(p-self.value)*w
        self.time=float(timestamp); return self.value.copy()
    def reset(self): self.value=self.time=None
