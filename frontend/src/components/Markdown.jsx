import React from "react";

/* --- Minimal markdown --------------------------------------------------- */
// Groq replies arrive as markdown (bold lead-ins, numbered steps). Rendering
// them as plain text put literal ** in the bubbles, so this walks the handful of
// constructs the assistant actually emits instead of pulling in a parser.

const INLINE_PATTERN = /(\*\*[^*]+\*\*|`[^`]+`)/g;

function renderInline(text, keyPrefix) {
  return text
    .split(INLINE_PATTERN)
    .filter(Boolean)
    .map((chunk, index) => {
      const key = `${keyPrefix}-${index}`;
      if (chunk.startsWith("**") && chunk.endsWith("**")) {
        return <strong key={key}>{chunk.slice(2, -2)}</strong>;
      }
      if (chunk.startsWith("`") && chunk.endsWith("`")) {
        return <code key={key}>{chunk.slice(1, -1)}</code>;
      }
      return <React.Fragment key={key}>{chunk}</React.Fragment>;
    });
}

export function renderMarkdown(text) {
  const blocks = [];
  let list = null;
  let para = [];

  function flushPara() {
    if (!para.length) return;
    const index = blocks.length;
    blocks.push(<p key={`p-${index}`}>{renderInline(para.join(" "), `p${index}`)}</p>);
    para = [];
  }

  function flushList() {
    if (!list) return;
    const index = blocks.length;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={`l-${index}`} className="md-list">
        {list.items.map((item, itemIndex) => (
          <li key={itemIndex}>{renderInline(item, `l${index}-${itemIndex}`)}</li>
        ))}
      </Tag>,
    );
    list = null;
  }

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();

    if (!line) {
      flushPara();
      flushList();
      continue;
    }

    const ordered = line.match(/^\d+[.)]\s+(.*)$/);
    const bullet = line.match(/^[-*\u2022]\s+(.*)$/);

    if (ordered) {
      flushPara();
      if (!list || !list.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(ordered[1]);
    } else if (bullet) {
      flushPara();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1]);
    } else if (list && /^\s/.test(rawLine)) {
      // Indented run-on belongs to the list item above it, not a new paragraph.
      list.items[list.items.length - 1] += ` ${line}`;
    } else {
      flushList();
      para.push(line);
    }
  }

  flushPara();
  flushList();
  return blocks;
}

