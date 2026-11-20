from collections import deque
import numpy as np

class FixationDetector:
    """Online dispersion detector. Event IDs are provisional while active; event finalizes on exit.

    Invalid samples break a candidate and finalize an active event. Duration is elapsed monotonic time.
    """
    def __init__(self,radius_px=70,min_duration_s=.1): self.radius=float(radius_px); self.minimum=float(min_duration_s); self.samples=deque(); self.next_id=1; self.active=None
    def update(self,timestamp,point,valid=True):
        p=np.asarray(point,float) if point is not None else np.array([np.nan,np.nan])
        if not valid or not np.isfinite(p).all():
            finished=self.active; self.active=None; self.samples.clear(); return {"id":None,"duration_s":0.0,"finalized":finished}
        self.samples.append((float(timestamp),p))
        while self.samples and np.linalg.norm(self.samples[-1][1]-self.samples[0][1])>self.radius: self.samples.popleft()
        start=self.samples[0][0]; duration=float(timestamp-start)
        if duration>=self.minimum and self.active is None: self.active=self.next_id; self.next_id+=1
        return {"id":self.active,"duration_s":duration if self.active else 0.0,"finalized":None}
    def finish(self):
        done=self.active; self.active=None; self.samples.clear(); return done
