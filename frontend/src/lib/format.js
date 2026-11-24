export const SCORE_BANDS = [
  { min: 80, label: "Excellent", tone: "strong" },
  { min: 65, label: "Good", tone: "good" },
  { min: 50, label: "Average", tone: "fair" },
  { min: -Infinity, label: "Needs Improvement", tone: "weak" },
];

export const ACCEPTED_TYPES = /^image\/(jpeg|png|webp|avif)$/;

export const COMPONENT_HELP = {
  "Attention Fit": "Does each element (product, headline, CTA, price, logo) get a healthy share of attention?",
  "Visual Hierarchy": "Is the viewing order right: product first, then headline / offer, logo last?",
  Focus: "How little attention is lost to background clutter, and no single element hogging everything.",
  Completeness: "Are the key ad elements present at all (product, headline, call-to-action, logo, price)?",
  Readability: "Text contrast against its background (WCAG 4.5:1 target) and text size.",
};

export function bandFor(score) {
  return SCORE_BANDS.find((band) => score >= band.min);
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(seconds) {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function downloadText(content, filename, mimeType) {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function attentionCsv(result) {
  const scores = result?.attention_scores_ms ?? {};
  const percentages = result?.pes?.attention_percentages ?? {};
  const predicted = result?.attention_comparison?.predicted ?? {};
  const rows = [["element", "attention_ms", "attention_percent", "saliency_estimate_percent"]];
  for (const label of Object.keys(percentages)) {
    rows.push([label, scores[label] ?? "", percentages[label] ?? 0, predicted[label] ?? ""]);
  }
  return rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
}

export function isGazeResult(result) {
  return String(result?.attention_source || "").startsWith("gaze") || String(result?.attention_source || "").startsWith("webgazer");
}
