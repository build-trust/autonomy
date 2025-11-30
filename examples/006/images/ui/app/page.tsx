"use client";

import { useState, useRef, useEffect } from "react";
import type { Message, StreamResponse } from "@/types/chat";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const appendCharByChar = async (text: string, delay = 5) => {
    for (const char of text) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      setMessages((prev) => {
        const newMsgs = [...prev];
        const lastMsg = newMsgs[newMsgs.length - 1];
        if (lastMsg && lastMsg.role === "assistant") {
          lastMsg.content += char;
        }
        return newMsgs;
      });
    }
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch("/api/agents/henry?stream=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader");

      const decoder = new TextDecoder();
      let buffer = "";
      let isFirstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            const parsed: StreamResponse = JSON.parse(line);
            for (const msg of parsed.messages || []) {
              if (msg?.content?.text) {
                if (isFirstChunk) {
                  isFirstChunk = false;
                  setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
                }
                await appendCharByChar(msg.content.text);
              }
            }
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error" }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "20px" }}>
      <h1>Chat with Henry</h1>
      <div
        ref={chatAreaRef}
        style={{
          border: "1px solid #ddd",
          borderRadius: "8px",
          padding: "20px",
          minHeight: "400px",
          marginBottom: "20px",
          overflowY: "auto",
        }}
      >
        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: "10px", textAlign: msg.role === "user" ? "right" : "left" }}>
            <div
              style={{
                display: "inline-block",
                padding: "8px 12px",
                borderRadius: "8px",
                background: msg.role === "user" ? "#007bff" : "#f0f0f0",
                color: msg.role === "user" ? "white" : "black",
                maxWidth: "70%",
              }}
            >
              {msg.content}
            </div>
          </div>
        ))}
      </div>
      <form onSubmit={sendMessage} style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          placeholder="Type a message..."
          style={{ flex: 1, padding: "10px", border: "1px solid #ddd", borderRadius: "4px" }}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          style={{
            padding: "10px 20px",
            background: loading || !input.trim() ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: "4px",
          }}
        >
          {loading ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
