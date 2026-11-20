import csv
from pathlib import Path

FIELDS=["timestamp_monotonic_s","elapsed_sample_ms","raw_x_px","raw_y_px","filtered_x_px","filtered_y_px","valid","quality_score","invalid_reason","fixation_id","fixation_duration_s","left_iris_x","left_iris_y","left_eye_height","right_iris_x","right_iris_y","right_eye_height","head_yaw","head_pitch","head_roll"]

class Recorder:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.file=self.path.open("w",newline=""); self.writer=csv.DictWriter(self.file,fieldnames=FIELDS); self.writer.writeheader(); self.last=None
    def write(self,row):
        t=float(row["timestamp_monotonic_s"]); row={k:row.get(k,"") for k in FIELDS}; row["elapsed_sample_ms"]="" if self.last is None else round((t-self.last)*1000,2); self.last=t; self.writer.writerow(row); self.file.flush()
    def close(self): self.file.close()
