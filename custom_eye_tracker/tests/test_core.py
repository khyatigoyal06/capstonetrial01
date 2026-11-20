import csv
from types import SimpleNamespace as LM
import numpy as np
import pytest

from custom_eye_tracker.calibration import TARGETS,TRAIN_IDX,VALID_IDX,TEST_IDX,robust_summary,target_pixel
from custom_eye_tracker.evaluate import select_smoothing_tau,summarize
from custom_eye_tracker.features import extract_features,valid_feature_vector
from custom_eye_tracker.fixation import FixationDetector
from custom_eye_tracker.gaze_model import RidgeGazeModel
from custom_eye_tracker.recorder import Recorder
from custom_eye_tracker.smoothing import ExponentialSmoother
from custom_eye_tracker.tracker import CustomEyeTracker

def landmarks():
    a=[LM(x=.5,y=.5) for _ in range(478)]
    # Anatomical eye corner and iris locations sufficient for deterministic geometry tests.
    for i,x,y in [(33,.3,.4),(133,.4,.4),(159,.35,.39),(145,.35,.41),(468,.35,.4),(469,.35,.4),(470,.35,.4),(471,.35,.4),(472,.35,.4),
                  (362,.6,.4),(263,.7,.4),(386,.65,.39),(374,.65,.41),(473,.65,.4),(474,.65,.4),(475,.65,.4),(476,.65,.4),(477,.65,.4)]: a[i]=LM(x=x,y=y)
    return a

def test_feature_normalization_and_invalid_eye():
    values,q=extract_features(landmarks()); assert values.shape==(6,); assert q["left_eye"] and q["right_eye"]; assert valid_feature_vector(values,q); assert values[0]==pytest.approx(.5)
    closed=landmarks()
    for i in (159,145,386,374): closed[i]=LM(x=closed[i].x,y=.4)
    blink,bq=extract_features(closed); assert bq["blink"] and bq["reason"]=="blink"; assert not valid_feature_vector(blink,bq)
    missing,_=extract_features([]); assert np.isnan(missing).all()

def test_robust_target_summary_and_coordinate_mapping():
    assert robust_summary([[1,3],[2,4],[100,5]])[0]==2
    assert target_pixel((.5,.5),(101,51))==(50,25)

def test_location_splits_are_disjoint_and_cover_screen():
    assert (len(TRAIN_IDX),len(VALID_IDX),len(TEST_IDX))==(15,5,5)
    assert not (set(TRAIN_IDX)&set(VALID_IDX) or set(TRAIN_IDX)&set(TEST_IDX) or set(VALID_IDX)&set(TEST_IDX))
    assert len(set(TRAIN_IDX)|set(VALID_IDX)|set(TEST_IDX))==len(TARGETS)==25
    assert {TARGETS[i] for i in TEST_IDX}=={(.08,.08),(.92,.08),(.5,.5),(.08,.92),(.92,.92)}

def test_ridge_roundtrip_and_metadata_guard(tmp_path):
    x=np.arange(60,dtype=float).reshape(10,6); y=np.c_[x[:,0]*2+4,x[:,1]*-1+9]
    m=RidgeGazeModel(1).fit(x,y); path=tmp_path/"model.json"; m.save(path,{"screen":"A"})
    loaded,meta=RidgeGazeModel.load(path,{"screen":"A"}); np.testing.assert_allclose(loaded.predict(x),m.predict(x)); assert meta=={"screen":"A"}
    assert loaded.fitted; assert loaded.predict(x[0]).shape==(2,); assert CustomEyeTracker(loaded).get_gaze(x[0],1.0).valid
    with pytest.raises(ValueError): RidgeGazeModel.load(path,{"screen":"B"})

def test_invalid_frame_and_timestamped_smoothing():
    q={"face":True,"blink":False}; x=np.array([1.,2.,.2,1.,2.,.2]); assert valid_feature_vector(x,q)
    q["blink"]=True; assert not valid_feature_vector(x,q)
    s=ExponentialSmoother(.1); np.testing.assert_allclose(s.update([0,0],1),[0,0]); assert s.update([float("nan"),1],1.1) is None
    assert np.linalg.norm(s.update([10,0],1.1))>0

def test_fixation_elapsed_time_finalization():
    d=FixationDetector(5,.2)
    assert d.update(1,[10,10])["id"] is None
    d.update(1.1,[11,10]); active=d.update(1.21,[10,11]); assert active["id"]==1; assert active["duration_s"]>=.2
    assert d.update(1.3,None,False)["finalized"]==1

def test_evaluation_and_csv_elapsed_duration(tmp_path):
    out=summarize([[0,0],[100,100]],[[3,4],[100,100]],[True,True]); assert out["mean_px"]==pytest.approx(2.5); assert out["locations"]==2
    r=Recorder(tmp_path/"samples.csv"); r.write({"timestamp_monotonic_s":1,"valid":True}); r.write({"timestamp_monotonic_s":1.08,"valid":False,"invalid_reason":"blink"}); r.close()
    with (tmp_path/"samples.csv").open() as h: rows=list(csv.DictReader(h))
    assert rows[0]["elapsed_sample_ms"]==""; assert rows[1]["elapsed_sample_ms"]=="80.0"; assert rows[1]["valid"]=="False"

def test_smoothing_selection_uses_grouped_validation_targets():
    class IdentityModel:
        def predict(self,features): return np.asarray(features,float)[:,:2]
    rows=[{"target_px":[10,20],"raw_samples":[{"timestamp_monotonic_s":1.0,"features":[10,20]},{"timestamp_monotonic_s":1.1,"features":[10,20]}]},
          {"target_px":[80,90],"raw_samples":[{"timestamp_monotonic_s":2.0,"features":[80,90]},{"timestamp_monotonic_s":2.1,"features":[80,90]}]}]
    tau,scores=select_smoothing_tau(IdentityModel(),rows,[0,.1])
    assert tau==0
    assert scores[0]["median_px"]==0
