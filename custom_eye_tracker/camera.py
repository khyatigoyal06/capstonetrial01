import time

class Camera:
    def __init__(self,index=0,width=1280,height=720):
        try: import cv2
        except ImportError as exc: raise RuntimeError("OpenCV is missing. Install custom_eye_tracker/requirements.txt in the virtual environment.") from exc
        self.cv2=cv2; self.cap=cv2.VideoCapture(index)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera index {index}. In macOS System Settings → Privacy & Security → Camera, allow the Terminal app running Python; restart Terminal after changing permission, or try --camera N.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); self.started=time.monotonic()
    def read(self):
        ok,frame=self.cap.read()
        if not ok or frame is None: return None
        return frame
    def close(self): self.cap.release()
