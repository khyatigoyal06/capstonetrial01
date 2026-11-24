import React, { useEffect, useState } from "react";
import { History, Trash2 } from "lucide-react";
import { bandFor, formatDate } from "../lib/format.js";

export default function HistoryPanel({ refreshKey, onOpen, activeId }) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  async function load() {
    try {
      const response = await fetch("/api/analyses?limit=20");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not load history.");
      setItems(payload.analyses || []);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => { load(); }, [refreshKey]);

  async function remove(id) {
    await fetch(`/api/analyses/${id}`, { method: "DELETE" });
    load();
  }

  if (!items.length && !error) return null;
  return (
    <section className="panel history-panel">
      <div className="panel-heading">
        <History size={18} />
        <h2>Recent analyses</h2>
        <span className="heading-note">saved on this computer</span>
      </div>
      {error && <p className="component-help">{error}</p>}
      <div className="history-list">
        {items.map((item) => {
          const tone = bandFor(item.score ?? 0).tone;
          return (
            <div key={item.analysis_id} className={`history-item${item.analysis_id === activeId ? " active" : ""}`}>
              <button type="button" className="history-open" onClick={() => onOpen(item.analysis_id)}>
                <span className={`history-score tone-text-${tone}`}>{Math.round(item.score ?? 0)}</span>
                <span className="history-name" title={item.filename}>{item.filename || "poster"}</span>
                <span className="history-meta">{String(item.kind).startsWith("gaze") || String(item.kind).startsWith("webgazer") ? "gaze" : "saliency"} · {formatDate(item.created_at)}</span>
              </button>
              <button type="button" className="history-delete" title="Delete" aria-label="Delete analysis" onClick={() => remove(item.analysis_id)}>
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}
