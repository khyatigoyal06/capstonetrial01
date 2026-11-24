import React, { useEffect, useRef, useState } from "react";
import { AlertCircle, Loader2, MessageSquare, Send } from "lucide-react";
import { renderMarkdown } from "./Markdown.jsx";

/* --- Chat --------------------------------------------------------------- */

const STARTERS = [
  "What is the single highest-impact change I should make?",
  "How do I make the CTA stand out more?",
  "Is the text readable enough? What would you change?",
];

export default function ChatPanel({ analysisId, chatReady, chatModel }) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    setMessages([]);
    setChatError("");
  }, [analysisId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;

    // Snapshot the prior turns: the server needs history without the message
    // it is about to answer, and setState is async.
    const history = messages.map(({ role, content }) => ({ role, content }));

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setDraft("");
    setSending(true);
    setChatError("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis_id: analysisId, message: text, history }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The assistant could not reply.");
      setMessages((prev) => [...prev, { role: "assistant", content: payload.reply }]);
    } catch (err) {
      setChatError(err.message);
    } finally {
      setSending(false);
    }
  }

  function useStarter(text) {
    setDraft(text);
    inputRef.current?.focus();
  }

  return (
    <div className="chat-panel">
      <div className="panel-heading">
        <MessageSquare size={18} />
        <h2>Ask about this poster</h2>
        {chatReady && <span className="chat-model">{chatModel}</span>}
      </div>

      {!chatReady ? (
        <div className="empty-state compact">
          Set <code>GROQ_API_KEY</code> in a <code>.env</code> file at the project root to enable
          the assistant.
        </div>
      ) : (
        <>
          <div className="chat-log" ref={scrollRef}>
            {messages.length === 0 && (
              <div className="chat-starters">
                <p>The assistant sees your poster and its scores. Try:</p>
                {STARTERS.map((text) => (
                  <button key={text} type="button" onClick={() => useStarter(text)}>
                    {text}
                  </button>
                ))}
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`bubble bubble-${message.role}`}>
                {message.role === "assistant" ? renderMarkdown(message.content) : message.content}
              </div>
            ))}

            {sending && (
              <div className="bubble bubble-assistant bubble-pending">
                <Loader2 className="spin" size={15} /> Thinking
              </div>
            )}
          </div>

          {chatError && (
            <div className="error-box">
              <AlertCircle size={17} />
              <span>{chatError}</span>
            </div>
          )}

          <form className="chat-composer" onSubmit={send}>
            <input
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask how to improve this poster"
              aria-label="Message the assistant"
            />
            <button className="secondary-button" type="submit" disabled={sending || !draft.trim()}>
              <Send size={16} />
              Send
            </button>
          </form>
        </>
      )}
    </div>
  );
}

