import React, { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Eye, Flame, ImageUp, Loader2, ScanEye, Target, X } from "lucide-react";
import HistoryPanel from "./components/HistoryPanel.jsx";
import ResultsView from "./components/ResultsView.jsx";
import { ACCEPTED_TYPES, formatBytes, isGazeResult } from "./lib/format.js";

export default function App() {
  const [activeMode, setActiveMode] = useState("poster");
  const [poster, setPoster] = useState(null);
  const [posterOrigin, setPosterOrigin] = useState("");
  const [studyAnnotations, setStudyAnnotations] = useState(null);
  const [gazeFile, setGazeFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [view, setView] = useState("boxes");
  const [health, setHealth] = useState({ chat_configured: false, chat_model: "" });
  const [historyKey, setHistoryKey] = useState(0);
  const studyFrameRef = useRef(null);
  const gazePosterRequestRef = useRef(0);
  const isGazeMode = activeMode === "gaze";

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ chat_configured: false, chat_model: "" }));
  }, []);

  // Carry the selected local dataset poster from the capture frame into gaze analysis.
  useEffect(() => {
    async function receiveStudyPoster(event) {
      if (
        event.origin !== window.location.origin ||
        event.source !== studyFrameRef.current?.contentWindow
      ) return;

      const { type, name, imageSrc, file: sharedFile, width, height, aois } = event.data ?? {};
      if (type !== "gaze-poster:selected" || typeof name !== "string") return;

      let source = null;
      if (!(sharedFile instanceof Blob)) {
        if (typeof imageSrc !== "string") return;
        try {
          source = new URL(imageSrc, window.location.origin);
        } catch {
          return;
        }
        if (
          source.origin !== window.location.origin ||
          !source.pathname.startsWith("/Capstone.yolov8/train/images/")
        ) return;
      }

      const requestId = ++gazePosterRequestRef.current;
      try {
        let file;
        if (sharedFile instanceof Blob) {
          const fileType = sharedFile.type || "image/jpeg";
          if (!ACCEPTED_TYPES.test(fileType)) throw new Error("This poster format cannot be analyzed.");
          file = new File([sharedFile], name, { type: fileType });
        } else {
          const response = await fetch(source.href);
          if (!response.ok) throw new Error("The selected poster could not be loaded.");
          const blob = await response.blob();
          const fileType = blob.type || "image/jpeg";
          if (!ACCEPTED_TYPES.test(fileType)) throw new Error("This poster format cannot be analyzed.");
          file = new File([blob], name, { type: fileType });
        }
        if (requestId !== gazePosterRequestRef.current) return;

        setPoster(file);
        setGazeFile(null);
        setPosterOrigin(name);
        setStudyAnnotations({
          width: Number(width) || 0,
          height: Number(height) || 0,
          aois: Array.isArray(aois) ? aois : [],
        });
        setResult(null);
        setError("");
        setView("saliency");
        setPreviewUrl((old) => {
          if (old) URL.revokeObjectURL(old);
          return URL.createObjectURL(file);
        });
      } catch (err) {
        if (requestId === gazePosterRequestRef.current) setError(err.message);
      }
    }

    window.addEventListener("message", receiveStudyPoster);
    return () => window.removeEventListener("message", receiveStudyPoster);
  }, []);

  // Object URLs leak until revoked, and a user may try many posters in a session.
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const acceptFile = useCallback((file) => {
    gazePosterRequestRef.current += 1;
    setPosterOrigin("");
    setStudyAnnotations(null);
    setResult(null);
    setError("");
    setView("boxes");
    setGazeFile(null);

    if (file && !ACCEPTED_TYPES.test(file.type)) {
      setPoster(null);
      setPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return "";
      });
      setError("That file is not a JPG, PNG, WEBP, or AVIF image.");
      return;
    }

    setPoster(file ?? null);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return file ? URL.createObjectURL(file) : "";
    });
  }, []);

  function handlePosterChange(event) {
    acceptFile(event.target.files?.[0]);
  }

  // The label only ever looked like a dropzone; these handlers make it behave
  // like one. A dropped File goes straight to state, so the hidden input stays
  // empty and is re-read on the next click.
  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function clearPoster(event) {
    event.preventDefault();
    event.stopPropagation();
    acceptFile(null);
  }

  async function analyzePoster() {
    if (!poster) {
      setError("Upload a poster image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", poster);
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/analyze-poster", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Analysis failed.");
      setResult(payload);
      setHistoryKey((key) => key + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function analyzeGaze() {
    if (!poster || !gazeFile) {
      setError("Choose both the poster image and its gaze CSV or TSV file.");
      return;
    }
    const formData = new FormData();
    formData.append("image_file", poster);
    formData.append("gaze_file", gazeFile);
    if (studyAnnotations?.aois?.length) {
      formData.append("aoi_json", JSON.stringify(studyAnnotations));
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch("/api/analyze-gaze", { method: "POST", body: formData });
      const responseText = await response.text();
      let payload;
      try {
        payload = JSON.parse(responseText);
      } catch {
        payload = { detail: responseText || "The server returned an empty response." };
      }
      if (!response.ok) throw new Error(payload.detail || "Gaze analysis failed.");
      setResult(payload);
      setView("saliency");
      setHistoryKey((key) => key + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function changeMode(mode) {
    setActiveMode(mode);
    setError("");
    setResult(null);
    setView(mode === "poster" ? "boxes" : "saliency");
  }

  async function openAnalysis(id) {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`/api/analyses/${id}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not open that analysis.");
      const gaze = isGazeResult(payload);
      setActiveMode(gaze ? "gaze" : "poster");
      setResult(payload);
      setView(gaze ? "saliency" : "boxes");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Food &amp; beverage poster analysis</p>
            <h1>Poster PES Analyzer</h1>
          </div>
          <div className="status-pill">
            <Eye size={15} />
            {activeMode === "study" ? "Live eye-tracking study" : isGazeMode ? "Gaze data analysis" : "YOLOv8 + saliency"}
          </div>
        </header>

        <nav className="mode-tabs" aria-label="Analysis mode">
          <button className={activeMode === "poster" ? "active" : ""} onClick={() => changeMode("poster")}>
            <Target size={16} /> Poster review
          </button>
          <button className={activeMode === "study" ? "active" : ""} onClick={() => changeMode("study")}>
            <Eye size={16} /> Live eye-tracking study
          </button>
          <button className={activeMode === "gaze" ? "active" : ""} onClick={() => changeMode("gaze")}>
            <ScanEye size={16} /> Analyze gaze file
          </button>
        </nav>

        {activeMode === "study" ? (
          <section className="study-panel panel">
            <div className="study-intro">
              <div>
                <p className="eyebrow">Research mode · webcam based</p>
                <h2>Collect a gaze session</h2>
                <p>Choose a poster, allow camera access, complete calibration and its accuracy check, then stop and download the CSV. Upload that CSV in Analyze gaze file to see its heatmap and metrics here.</p>
              </div>
              <span className="privacy-note"><Eye size={15} /> Camera frames are processed by this app's local API</span>
            </div>
            <iframe
              ref={studyFrameRef}
              className="gaze-frame"
              src="/api/gaze/capture"
              title="Poster eye-tracking study"
              allow="camera; autoplay; fullscreen"
              allowFullScreen
              onLoad={(event) => event.currentTarget.contentWindow?.postMessage({ type: "gaze-poster:request" }, window.location.origin)}
            />
            <p className="study-footnote">The capture page is served from this project. Your browser will ask before enabling the webcam. The gaze engine calibrates on 13 dots by dwell (no clicking), measures its own accuracy on 8 separate dots, blocks recording when accuracy is poor, and re-checks drift every 30 s. The CSV is analyzed only after you choose to upload it.</p>
          </section>
        ) : (
        <>
        <div className="layout">
          <section className="panel uploader-panel">
            <div className="panel-heading">
              <ImageUp size={18} />
              <h2>{isGazeMode ? "Poster for AOI mapping" : "Upload Poster"}</h2>
            </div>
            {isGazeMode && posterOrigin && (
              <p className="poster-source-note">
                Matched from the live study: <strong>{posterOrigin}</strong>. Choose a different poster below if needed.
              </p>
            )}

            <label
              className={`dropzone${dragging ? " dragging" : ""}${previewUrl ? " filled" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input type="file" accept="image/*" onChange={handlePosterChange} />
              {previewUrl ? (
                <>
                  <img src={previewUrl} alt="Uploaded poster preview" />
                  <button
                    type="button"
                    className="clear-button"
                    onClick={clearPoster}
                    aria-label="Remove poster"
                  >
                    <X size={15} />
                  </button>
                </>
              ) : (
                <span className="dropzone-hint">
                  <ImageUp size={26} />
                  <strong>Drop a poster here</strong>
                  JPG, PNG, WEBP, or AVIF &mdash; or click to browse
                </span>
              )}
            </label>

            {poster && (
              <p className="file-chip">
                <span title={poster.name}>{poster.name}</span>
                <span className="file-size">{formatBytes(poster.size)}</span>
              </p>
            )}

            {isGazeMode && (
              <label className="gaze-file-picker">
                <span className="eyebrow">Gaze export</span>
                <input type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" onChange={(event) => setGazeFile(event.target.files?.[0] ?? null)} />
                <small>{gazeFile ? `${gazeFile.name} · ${formatBytes(gazeFile.size)}` : "Choose the CSV or TSV recorded for this poster"}</small>
              </label>
            )}

            <button className="primary-button" onClick={isGazeMode ? analyzeGaze : analyzePoster} disabled={loading || (isGazeMode && (!poster || !gazeFile))}>
              {loading ? <Loader2 className="spin" size={17} /> : <Target size={17} />}
              {loading ? "Analyzing" : isGazeMode ? "Analyze gaze session" : "Analyze Poster"}
            </button>

            {error && (
              <div className="error-box">
                <AlertCircle size={17} />
                <span>{error}</span>
              </div>
            )}
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <Target size={18} />
              <h2>Detection Output</h2>
              {result && (
                <div className="view-toggle">
                  <button
                    type="button"
                    className={view === "boxes" ? "active" : ""}
                    onClick={() => setView("boxes")}
                  >
                    <Target size={14} /> Boxes
                  </button>
                  <button
                    type="button"
                    className={view === "saliency" ? "active" : ""}
                    onClick={() => setView("saliency")}
                  >
                    <Flame size={14} /> Attention
                  </button>
                </div>
              )}
            </div>

            {result ? (
              <>
                <img
                  className="annotated-image"
                  src={view === "boxes" ? result.annotated_image : result.saliency_image}
                  alt={view === "boxes" ? "YOLO bounding boxes" : isGazeMode ? "Gaze heatmap" : "Estimated attention heatmap"}
                />
                <a className="download-link" href={view === "boxes" ? result.annotated_image : result.saliency_image} download={isGazeMode ? "gaze-heatmap.png" : "poster-attention.png"}>
                  Download {view === "boxes" ? "AOI preview" : "heatmap"}
                </a>
              </>
            ) : (
              <div className="empty-state">
                {loading ? (
                  <span className="loading-state">
                    <Loader2 className="spin" size={22} />
                    Detecting elements and estimating attention
                  </span>
                ) : (
                  isGazeMode ? "Upload the matching poster and gaze export to see measured attention." : "Bounding boxes appear here after analysis."
                )}
              </div>
            )}
          </section>
        </div>

        {result && <ResultsView result={result} health={health} />}
        </>
        )}
        {activeMode !== "study" && (
          <HistoryPanel refreshKey={historyKey} onOpen={openAnalysis} activeId={result?.analysis_id} />
        )}
      </section>
    </main>
  );
}

