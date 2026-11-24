import React, { useMemo } from "react";
import { FileText, Lightbulb, Sparkles, Type } from "lucide-react";
import ChatPanel from "./ChatPanel.jsx";
import { AttentionBar, ComponentBar, ScoreRing } from "./Meters.jsx";
import { attentionCsv, bandFor, downloadText, isGazeResult } from "../lib/format.js";

function Recommendations({ items }) {
  if (!items?.length) return null;
  return (
    <section className="panel recommendations-panel">
      <div className="panel-heading">
        <Lightbulb size={18} />
        <h2>How to improve this poster</h2>
        <span className="heading-note">highest impact first</span>
      </div>
      <ol className="rec-cards">
        {items.map((item, index) => {
          const rec = typeof item === "string" ? { title: item, detail: "", impact: 0 } : item;
          return (
            <li key={`${rec.title}-${index}`} className="rec-card">
              <span className="rec-index">{index + 1}</span>
              <div>
                <strong>{rec.title}</strong>
                {rec.detail && <p>{rec.detail}</p>}
                {rec.component && <span className="rec-component">{rec.component}</span>}
              </div>
              {rec.impact > 0 && <span className="rec-impact" title="PES points this change could win back">+{rec.impact.toFixed(1)}</span>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Readability({ items }) {
  if (!items?.length) return null;
  return (
    <section className="panel">
      <div className="panel-heading">
        <Type size={18} />
        <h2>Text readability</h2>
        <span className="heading-note">WCAG asks for 4.5:1 (3:1 for large text)</span>
      </div>
      <div className="table readability-table">
        <div className="table-row table-head"><span>Element</span><span>Contrast</span><span>Height</span><span>Score</span></div>
        {items.map((item) => (
          <div className="table-row" key={item.label}>
            <span>{item.label}</span>
            <span className={item.contrast_ratio >= 4.5 ? "ok-text" : item.contrast_ratio >= 3 ? "warn-text" : "bad-text"}>
              {item.contrast_ratio.toFixed(1)}:1
            </span>
            <span className="mono">{(item.height_share * 100).toFixed(1)}%</span>
            <strong>{Math.round(item.score)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function GazeSummary({ result }) {
  const stats = result.gaze_stats || {};
  const quality = result.gaze_quality || {};
  const comparison = result.attention_comparison || {};
  return (
    <section className="panel gaze-summary">
      <div>
        <p className="eyebrow">Session coverage</p>
        <strong>{result.gaze_sample_count} gaze samples mapped</strong>
      </div>
      {stats.fixations != null && <span>{stats.fixations} fixations · {((stats.fixation_time_ms || 0) / 1000).toFixed(1)} s</span>}
      {stats.off_poster_ms > 0 && <span>{(stats.off_poster_ms / 1000).toFixed(1)} s off the poster</span>}
      {quality.validation_median_error_screen_px != null && (
        <span className={`quality-chip quality-${quality.session_quality || "unknown"}`}>
          Accuracy {Math.round(quality.validation_median_error_screen_px)} px median{quality.session_quality ? ` · ${quality.session_quality}` : ""}
        </span>
      )}
      {quality.unreliable_aois?.length > 0 && (
        <span className="quality-chip quality-fair" title="Smaller than the measured gaze error, so their share is approximate.">
          Approximate: {quality.unreliable_aois.join(", ")}
        </span>
      )}
      {comparison.pearson_r != null && (
        <span className="quality-chip" title="Agreement between the saliency estimate (Poster Review) and this measured session">
          Saliency vs gaze: r = {comparison.pearson_r.toFixed(2)}, error {comparison.mean_abs_error_pct?.toFixed(1)} pts
        </span>
      )}
      <button className="secondary-button" onClick={() => downloadText(JSON.stringify(result, null, 2), "gaze-analysis.json", "application/json")}>Report JSON</button>
      <button className="secondary-button" onClick={() => downloadText(attentionCsv(result), "gaze-attention.csv", "text/csv")}>Attention CSV</button>
    </section>
  );
}

export default function ResultsView({ result, health }) {
  const pes = result.pes || {};
  const gaze = isGazeResult(result);
  const band = bandFor(pes.score ?? 0);
  const bands = pes.elements?.attention || {};
  const predicted = result.attention_comparison?.predicted;

  const attention = useMemo(
    () => Object.entries(pes.attention_percentages ?? {}).sort((a, b) => b[1] - a[1]),
    [pes.attention_percentages],
  );
  const weakest = useMemo(() => {
    const entries = Object.entries(pes.components ?? {});
    return entries.length ? entries.sort((a, b) => a[1] - b[1])[0] : null;
  }, [pes.components]);

  return (
    <>
      <section className="insights-grid">
        <article className="score-panel">
          <p className="eyebrow">Poster Effectiveness Score</p>
          <ScoreRing score={pes.score ?? 0} tone={band.tone} />
          <span className={`band band-${band.tone}`}>{pes.category}</span>
          {weakest && <p className="score-note">Weakest: <strong>{weakest[0]}</strong> at {Math.round(weakest[1])}</p>}
          <p className="score-note">
            Attention {gaze ? "measured with eye tracking" : "estimated from visual saliency"}. PES v{pes.version ?? 1} is a design heuristic, not a sales prediction.
          </p>
          <a className="download-link" href={`/api/analyses/${result.analysis_id}/report.pdf`}>
            <FileText size={14} /> Download PDF report
          </a>
        </article>

        <article className="components-panel">
          <p className="eyebrow">Component breakdown</p>
          {Object.entries(pes.components ?? {}).map(([name, value]) => (
            <ComponentBar key={name} name={name} value={value} weight={pes.weights?.[name] ?? 0} />
          ))}
          {pes.not_measured?.length > 0 && (
            <p className="component-help">Not measured for this poster: {pes.not_measured.join(", ")} (weights rescaled).</p>
          )}
        </article>

        <article className="attention-panel">
          <p className="eyebrow">{gaze ? "Measured attention share" : "Estimated attention share"}</p>
          {attention.map(([label, percent]) => (
            <AttentionBar key={label} label={label} percent={percent}
              band={bands[label]?.ideal_range}
              predicted={gaze && predicted ? predicted[label] : undefined} />
          ))}
          <p className="component-help">
            Green zone = healthy range.{gaze && predicted ? " White tick = what the saliency model predicted for the same element." : ""}
          </p>
        </article>
      </section>

      {gaze && <GazeSummary result={result} />}

      <Recommendations items={result.recommendations} />
      <Readability items={pes.elements?.text} />

      <section className="details">
        <div className="detections">
          <h2>Detected elements</h2>
          {!result.detections?.length ? (
            <div className="empty-state compact">The detector found no poster elements above the confidence threshold.</div>
          ) : (
            <div className="table">
              <div className="table-row table-head"><span>Element</span><span>Conf.</span><span>Position</span></div>
              {result.detections.map((item, index) => (
                <div className="table-row" key={`${item.class_name}-${index}`}>
                  <span>{item.class_name}</span>
                  <span className="mono">{Math.round(item.confidence * 100)}%</span>
                  <span className="dim">{item.relative_position}</span>
                </div>
              ))}
            </div>
          )}
          <h2 className="stacked-heading"><Sparkles size={16} /> Findings</h2>
          <ul className="insight-list">
            {(pes.insights ?? []).map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
        <div className="suggestions">
          <ChatPanel analysisId={result.analysis_id} chatReady={health.chat_configured} chatModel={health.chat_model} />
        </div>
      </section>
    </>
  );
}
