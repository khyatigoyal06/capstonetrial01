from __future__ import annotations
import argparse, json, random, time
from datetime import datetime, timezone
from pathlib import Path
from . import __version__
from .config import TrackerConfig

def run(args):
    import sys
    if sys.version_info < (3,9) or sys.version_info >= (3,13):
        print(f"BLOCKER: Python {sys.version_info.major}.{sys.version_info.minor} is outside MediaPipe's published Python 3.9–3.12 support. Create the documented Python 3.11 virtual environment."); return 2
    try:
        import numpy as np
    except ImportError:
        print("BLOCKER: NumPy is missing. Activate the custom_eye_tracker Python 3.11 virtual environment and install its requirements.txt."); return 2
    from .calibration import TARGETS,TRAIN_IDX,VALID_IDX,TEST_IDX,robust_summary,target_pixel
    from .camera import Camera
    from .evaluate import select_smoothing_tau,summarize
    from .features import extract_features,valid_feature_vector
    from .fixation import FixationDetector
    from .gaze_model import RidgeGazeModel,select_alpha
    from .landmarks import FaceLandmarker
    from .recorder import Recorder
    from .smoothing import ExponentialSmoother
    cfg=TrackerConfig(camera_index=args.camera,model_path=Path(args.model),output_dir=Path(args.output))
    try: camera=Camera(cfg.camera_index,cfg.camera_width,cfg.camera_height)
    except RuntimeError as e: print(f"BLOCKER: {e}"); return 2
    cv2=camera.cv2
    try: landmarker=FaceLandmarker(cfg.model_path)
    except RuntimeError as e: camera.close(); print(f"BLOCKER: {e}"); return 2
    try:
        cv2.namedWindow("Standalone Eye Tracker",cv2.WINDOW_NORMAL); cv2.setWindowProperty("Standalone Eye Tracker",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Standalone Eye Tracker",np.zeros((cfg.camera_height,cfg.camera_width,3),dtype=np.uint8)); cv2.waitKey(150); time.sleep(.1)
    except cv2.error as e:
        landmarker.close(); camera.close(); print(f"BLOCKER: Could not create a full-screen OpenCV window. Run from a logged-in macOS desktop session. {e}"); return 2
    # Coordinates refer to this OpenCV drawable surface. All targets and predictions
    # use its width/height; this avoids mixing camera, browser and Retina logical pixels.
    try: _,_,sw,sh=cv2.getWindowImageRect("Standalone Eye Tracker")
    except cv2.error as e:
        landmarker.close(); camera.close(); print(f"BLOCKER: Could not read drawable size from OpenCV. {e}"); return 2
    if sw<320 or sh<240:
        landmarker.close(); camera.close(); cv2.destroyAllWindows(); print(f"BLOCKER: OpenCV reported invalid fullscreen drawable size {sw}x{sh}; cannot map target and gaze coordinates safely."); return 2
    screen={"coordinate_system":"OpenCV fullscreen drawable pixels","display_selection":"macOS primary/default display at launch","width_px":sw,"height_px":sh,"camera_index":cfg.camera_index,"camera_requested_size":[cfg.camera_width,cfg.camera_height]}
    if args.diagnostics:
        start=time.monotonic(); previous=start
        try:
            while True:
                frame=camera.read()
                if frame is None: continue
                now=time.monotonic(); lms,_=landmarker.detect(frame,int((now-start)*1000)); _,q=extract_features(lms)
                view=cv2.flip(cv2.resize(frame,(sw,sh)),1)
                for lm in lms: cv2.circle(view,(int((1-lm.x)*(sw-1)),int(lm.y*(sh-1))),1,(255,100,40),-1)
                fps=1/max(now-previous,1e-6); previous=now
                lines=[f"Frames processed on this Mac | face={'yes' if q['face'] else 'no'} | left eye={'yes' if q['left_eye'] else 'no'} | right eye={'yes' if q['right_eye'] else 'no'} | blink={q['blink']}",f"MediaPipe may send performance/usage metrics | quality heuristic={q['landmark_quality']:.2f} | {fps:.1f} fps | q quit"]
                for i,line in enumerate(lines): cv2.putText(view,line,(20,35+i*30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow("Standalone Eye Tracker",view)
                if cv2.waitKey(1)&0xff==ord('q'): return 0
        finally:
            landmarker.close(); camera.close(); cv2.destroyAllWindows()
    run_dir=cfg.output_dir/f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True,exist_ok=False)
    records=[]; target_rows=[]; frame_n=0; started=time.monotonic(); last_draw=started; message="Local processing only: frames stay on this machine. Press q to quit."
    model=None; alpha_info={}; recorder=None; smoother=ExponentialSmoother(cfg.smoothing_tau_s); fix=FixationDetector(cfg.fixation_radius_px,cfg.fixation_min_duration_s)
    def collect(indices,phase):
        nonlocal message,last_draw
        order=list(indices); random.Random(431+len(indices)).shuffle(order); gathered=[]
        for ix in order:
            ratio=TARGETS[ix]; xy=target_pixel(ratio,(sw,sh)); valid=[]; raw=[]; attempted=0; begin=time.monotonic(); capture_at=begin+cfg.settle_s
            while time.monotonic()-begin<cfg.settle_s+cfg.sample_window_s:
                frame=camera.read()
                if frame is None: continue
                now=time.monotonic(); lms,result=landmarker.detect(frame,int((now-started)*1000)); feat,q=extract_features(lms)
                view=cv2.resize(frame,(sw,sh)); view=cv2.flip(view,1)
                # Flip camera view for natural selfie display; prediction landmarks/features
                # come from unflipped frame consistently.
                cv2.circle(view,xy,14,(0,170,255) if phase=="test" else (40,230,90),-1); cv2.circle(view,xy,4,(255,255,255),-1)
                for lm in lms: cv2.circle(view,(int((1-lm.x)*(sw-1)),int(lm.y*(sh-1))),1,(255,100,40),-1)
                if now>=capture_at: attempted+=1
                good=now>=capture_at and valid_feature_vector(feat,q)
                if good: valid.append(feat); raw.append((now,feat.copy()))
                progress="settling" if now<capture_at else f"window {min(100,int(100*(now-capture_at)/cfg.sample_window_s))}%"
                status=f"{phase.upper()} target {len(gathered)+1}/{len(order)} | {progress} | {len(valid)} valid | face {'YES' if q['face'] else 'NO'} eyes L/R {int(q['left_eye'])}/{int(q['right_eye'])} blink {q['blink']} | {1/max(now-last_draw,1e-6):.1f} fps"
                last_draw=now
                cv2.putText(view,status,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(view,"Settle, then look at dot. q quits; Space skips this target.",(20,65),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow("Standalone Eye Tracker",view); key=cv2.waitKey(1)&0xff
                if key==ord('q'): return None
                if key==32: break
            if len(valid)<cfg.min_samples:
                message=f"Target skipped: only {len(valid)} valid frames (need {cfg.min_samples}). Repeat run for a complete split."; print(message); continue
            summary=robust_summary(valid); gathered.append((ix,summary,xy,len(valid),attempted,raw))
            target_rows.append({"phase":phase,"target_index":ix,"target_ratio":ratio,"target_px":xy,"capture_duration_s":max(time.monotonic()-capture_at,0.0),"n_valid_frames":len(valid),"n_attempted_frames":attempted,"n_rejected_frames":attempted-len(valid),"feature_median":summary.tolist(),"raw_samples":[{"timestamp_monotonic_s":t,"features":f.tolist()} for t,f in raw]})
            (run_dir/"calibration_samples.json").write_text(json.dumps({"screen":screen,"targets":target_rows},indent=2))
        return gathered
    try:
        for phase,idx in (("train",TRAIN_IDX),("validation",VALID_IDX)):
            got=collect(idx,phase)
            if got is None: return 0
            records.extend(got)
        train=[x for x in records if x[0] in TRAIN_IDX]; val=[x for x in records if x[0] in VALID_IDX]
        if len(train)<8 or len(val)<len(VALID_IDX): print("BLOCKER: incomplete train or model-selection validation locations; restart and keep face/eyes visible. No final test was opened."); return 2
        model,alpha_info=select_alpha([r[1] for r in train],[r[2] for r in train],[r[1] for r in val],[r[2] for r in val])
        validation_rows=[r for r in target_rows if r["phase"]=="validation"]
        selected_smoothing_tau,smoothing_scores=select_smoothing_tau(model,validation_rows,args.smoothing_candidates,args.thresholds)
        smoother=ExponentialSmoother(selected_smoothing_tau)
        validation_predictions=[np.median(model.predict([s["features"] for s in row["raw_samples"]]),axis=0) for row in validation_rows]
        validation_raw=summarize([r["target_px"] for r in validation_rows],validation_predictions,[True]*len(validation_rows),args.thresholds)
        validation={"raw":validation_raw,"filtered":smoothing_scores[selected_smoothing_tau],"selected_smoothing_tau_s":selected_smoothing_tau}
        # Lock model and settings before opening the final test phase.
        meta={"tracker_version":__version__,"created_utc":datetime.now(timezone.utc).isoformat(),"screen":screen,"feature_names":["left/right iris normalized x/y and eye openness"],"train_locations":len(train),"validation_locations":len(val),"test_locations_planned":len(TEST_IDX),"split":"fixed target indices in calibration.py","validation":validation,"selection":{"ridge":alpha_info,"smoothing_median_error_by_tau":{str(k):v["median_px"] for k,v in smoothing_scores.items()}},"config":cfg.metadata()}
        model.save(run_dir/"calibration_model.json",meta)
        print(json.dumps({"validation":validation,"train_locations":len(train),"validation_locations":len(val),"selected_alpha":model.alpha,"selected_smoothing_tau_s":selected_smoothing_tau,"run_dir":str(run_dir)},indent=2))
        tests=collect(TEST_IDX,"test")
        if tests is None: return 0
        if len(tests)<len(TEST_IDX): print("BLOCKER: incomplete final test locations; no final accuracy result reported."); return 2
        test_raw=[]; test_filtered=[]; targets=[]; cover=[]; latencies=[]; attempted=0; valid_frames=0; capture_seconds=0.0; observed_rates=[]
        for ix,feature,xy,nv,nraw,samples in tests:
            # Report per-location median prediction. Frames remain nested within
            # their target and never inflate the independent location count.
            loc_raw=[]; loc_filter=ExponentialSmoother(selected_smoothing_tau); loc_filtered=[]; loc_latency=[]
            attempted+=nraw; valid_frames+=nv
            capture_seconds+=next((r["capture_duration_s"] for r in target_rows if r["phase"]=="test" and r["target_index"]==ix),0.0)
            if len(samples)>1: observed_rates.extend([1/(samples[j][0]-samples[j-1][0]) for j in range(1,len(samples)) if samples[j][0]>samples[j-1][0]])
            for ts,sample in samples:
                pt=time.monotonic(); pred=model.predict([sample])[0]; loc_latency.append((time.monotonic()-pt)*1000); loc_raw.append(pred); loc_filtered.append(loc_filter.update(pred,ts))
            targets.append(xy); test_raw.append(np.median(loc_raw,axis=0)); test_filtered.append(np.median(loc_filtered,axis=0)); cover.append(bool(nv))
            latencies.extend(loc_latency)
        final={"raw":summarize(targets,test_raw,cover,args.thresholds),"filtered":summarize(targets,test_filtered,cover,args.thresholds),"independent_test_locations":len(tests),"selected_smoothing_tau_s":selected_smoothing_tau,"valid_frame_coverage":valid_frames/max(attempted,1),"dropped_or_invalid_frame_rate":1-valid_frames/max(attempted,1),"ridge_regression_latency_ms_median":float(np.median(latencies)) if latencies else None,"ridge_regression_latency_ms_p90":float(np.percentile(latencies,90)) if latencies else None,"observed_camera_fps":attempted/capture_seconds if capture_seconds>0 else None,"observed_valid_sample_fps_median":float(np.median(observed_rates)) if observed_rates else None,"final_test_used_for_tuning":False,"note":"latency covers only the ridge regression call, not camera capture or landmark inference; final locations were excluded from training and model selection; five-location estimate"}
        (run_dir/"calibration_samples.json").write_text(json.dumps({"screen":screen,"targets":target_rows},indent=2))
        (run_dir/"evaluation.json").write_text(json.dumps({"metadata":meta,"validation":validation,"final_test":final},indent=2))
        print("FINAL HELD-OUT TEST (do not tune from these results):\n"+json.dumps(final,indent=2))
        print("Live prediction and recording: press r to toggle CSV recording, q to quit.")
        while True:
            t0=time.monotonic(); frame=camera.read()
            if frame is None: continue
            lms,result=landmarker.detect(frame,int((t0-started)*1000)); feature,q=extract_features(lms); valid=valid_feature_vector(feature,q); prediction=model.predict([feature])[0] if valid else None
            filtered=smoother.update(prediction,t0) if valid else None
            if not valid: smoother.reset()
            fx=fix.update(t0,filtered,valid)
            view=cv2.resize(frame,(sw,sh)); view=cv2.flip(view,1)
            for lm in lms: cv2.circle(view,(int((1-lm.x)*(sw-1)),int(lm.y*(sh-1))),1,(255,100,40),-1)
            if valid:
                px=(int(np.clip(filtered[0],0,sw-1)),int(np.clip(filtered[1],0,sh-1))); cv2.circle(view,px,10,(0,255,255),-1)
            fps=1/max(time.monotonic()-t0,1e-6); frame_n+=1
            text=f"face={'yes' if q['face'] else 'no'} L={q['left_eye']} R={q['right_eye']} blink={q['blink']} valid={valid}  {fps:.1f} fps"
            cv2.putText(view,text,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(view,"Frames processed here; MediaPipe may send usage metrics | r record | q quit",(20,65),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow("Standalone Eye Tracker",view); key=cv2.waitKey(1)&0xff
            if key==ord('r'):
                if recorder: recorder.close(); recorder=None; print("Recording saved.")
                else: recorder=Recorder(cfg.output_dir/f"gaze_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"); print(f"Recording to {recorder.path}")
            if recorder:
                row={"timestamp_monotonic_s":t0,"raw_x_px":prediction[0] if valid else "","raw_y_px":prediction[1] if valid else "","filtered_x_px":filtered[0] if valid else "","filtered_y_px":filtered[1] if valid else "","valid":valid,"quality_score":q["landmark_quality"],"invalid_reason":q["reason"],"fixation_id":fx["id"] or "","fixation_duration_s":fx["duration_s"],**dict(zip(("left_iris_x","left_iris_y","left_eye_height","right_iris_x","right_iris_y","right_eye_height"),feature))}
                recorder.write(row)
            if key==ord('q'): break
        return 0
    finally:
        if recorder: recorder.close()
        landmarker.close(); camera.close(); cv2.destroyAllWindows()

def main():
    p=argparse.ArgumentParser(description="Standalone local webcam gaze tracker")
    p.add_argument("--camera",type=int,default=0); p.add_argument("--model",default="custom_eye_tracker/models/face_landmarker.task"); p.add_argument("--output",default="records"); p.add_argument("--thresholds",type=float,nargs="+",default=[50,100,150],help="Pixel-error thresholds reported in evaluation"); p.add_argument("--smoothing-candidates",type=float,nargs="+",default=[0.0,0.05,0.12,0.2],help="Smoother time constants tested on validation locations")
    p.add_argument("--diagnostics",action="store_true",help="Preview face/eye landmarks, quality status and frame rate without calibration")
    args=p.parse_args(); raise SystemExit(run(args))
if __name__=="__main__": main()
