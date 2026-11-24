import React from "react";
import { bandFor, COMPONENT_HELP } from "../lib/format.js";

export function ScoreRing({ score, tone }) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const dash = (Math.max(0, Math.min(100, score)) / 100) * circumference;
  return (
    <div className="score-ring">
      <svg viewBox="0 0 128 128" aria-hidden="true">
        <circle className="ring-track" cx="64" cy="64" r={radius} />
        <circle className={`ring-value stroke-${tone}`} cx="64" cy="64" r={radius}
          strokeDasharray={`${dash} ${circumference}`} transform="rotate(-90 64 64)" />
      </svg>
      <div className="score-ring-center">
        <strong>{Math.round(score)}</strong>
        <span>/ 100</span>
      </div>
    </div>
  );
}

export function ComponentBar({ name, value, weight }) {
  const tone = bandFor(value).tone;
  return (
    <div className="component-row" title={COMPONENT_HELP[name] || ""}>
      <div className="component-label">
        <span>{name}</span>
        <span className="component-weight">{Math.round(weight)}%</span>
      </div>
      <div className="meter" role="img" aria-label={`${name}: ${Math.round(value)} out of 100`}>
        <div className={`meter-fill tone-${tone}`} style={{ width: `${Math.max(2, value)}%` }} />
      </div>
      <strong className="component-value">{Math.round(value)}</strong>
      {COMPONENT_HELP[name] && <p className="component-help">{COMPONENT_HELP[name]}</p>}
    </div>
  );
}

export function AttentionBar({ label, percent, predicted, band }) {
  const muted = label === "Background";
  const inBand = band && percent >= band[0] && percent <= band[1];
  return (
    <div className="attention-row">
      <span className="attention-label">{label}</span>
      <div className="meter attention-meter" role="img" aria-label={`${label}: ${percent.toFixed(1)} percent`}>
        {band && <div className="meter-band" style={{ left: `${band[0]}%`, width: `${band[1] - band[0]}%` }} title={`Healthy range ${band[0]}-${band[1]}%`} />}
        <div className={muted ? "meter-fill tone-muted" : `meter-fill ${band ? (inBand ? "tone-good" : "tone-fair") : "tone-accent"}`}
          style={{ width: `${Math.max(1, percent)}%` }} />
        {typeof predicted === "number" && (
          <div className="meter-marker" style={{ left: `${Math.min(100, predicted)}%` }} title={`Saliency estimate ${predicted.toFixed(1)}%`} />
        )}
      </div>
      <strong className="attention-value">{percent.toFixed(1)}%</strong>
    </div>
  );
}
