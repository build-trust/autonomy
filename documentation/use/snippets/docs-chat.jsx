export const DocsChat = () => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hey! I'm your interface to a team of agents running on Autonomy. Ask me anything about the documentation!",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [visitorId] = useState(() => Math.random().toString(36).slice(2));
  const [conversationId] = useState(() => Math.random().toString(36).slice(2));
  const chatAreaRef = useRef(null);

  const DOCS_API_BASE = "https://a9eb812238f753132652ae09963a05e9-docs.cluster.autonomy.computer";

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${DOCS_API_BASE}/agents/docs?stream=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          scope: visitorId,
          conversation: conversationId,
        }),
      });

      if (!response.ok) throw new Error("Failed to get response");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      let buffer = "";
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim() || !line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.text) {
              fullText += parsed.text;
              const currentText = fullText;
              setMessages((prev) => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
                  updated[lastIndex] = { ...updated[lastIndex], content: currentText };
                }
                return updated;
              });
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error. Please try again." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const styles = {
    container: {
      border: "1px solid var(--border-color, #e5e7eb)",
      borderRadius: "12px",
      overflow: "hidden",
      backgroundColor: "var(--background-color, #fff)",
      maxWidth: "100%",
    },
    chatArea: {
      height: "400px",
      overflowY: "auto",
      padding: "16px",
      display: "flex",
      flexDirection: "column",
      gap: "12px",
      backgroundColor: "var(--background-muted, #f9fafb)",
    },
    messageRow: {
      display: "flex",
      gap: "8px",
      alignItems: "flex-start",
    },
    userRow: {
      justifyContent: "flex-end",
    },
    avatar: {
      width: "28px",
      height: "28px",
      borderRadius: "50%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: "12px",
      flexShrink: 0,
    },
    assistantAvatar: {
      backgroundColor: "#f54a00",
      color: "white",
    },
    userAvatar: {
      backgroundColor: "var(--background-muted, #e5e7eb)",
      color: "var(--text-color, #374151)",
    },
    bubble: {
      maxWidth: "80%",
      padding: "10px 14px",
      borderRadius: "16px",
      fontSize: "14px",
      lineHeight: "1.5",
    },
    assistantBubble: {
      backgroundColor: "var(--background-color, #fff)",
      border: "1px solid var(--border-color, #e5e7eb)",
      color: "var(--text-color, #374151)",
    },
    userBubble: {
      backgroundColor: "#f54a00",
      color: "white",
    },
    inputArea: {
      borderTop: "1px solid var(--border-color, #e5e7eb)",
      padding: "12px 16px",
      backgroundColor: "var(--background-color, #fff)",
    },
    form: {
      display: "flex",
      gap: "8px",
    },
    input: {
      flex: 1,
      padding: "10px 14px",
      borderRadius: "8px",
      border: "1px solid var(--border-color, #e5e7eb)",
      fontSize: "14px",
      backgroundColor: "var(--background-color, #fff)",
      color: "var(--text-color, #374151)",
      outline: "none",
    },
    button: {
      padding: "10px 16px",
      borderRadius: "8px",
      border: "none",
      backgroundColor: "#f54a00",
      color: "white",
      fontSize: "14px",
      fontWeight: "500",
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      gap: "6px",
    },
    buttonDisabled: {
      opacity: 0.5,
      cursor: "not-allowed",
    },
    thinking: {
      color: "var(--text-muted, #9ca3af)",
    },
  };

  const ThinkingIndicator = () => {
    const [dots, setDots] = useState("");
    useEffect(() => {
      const interval = setInterval(() => {
        setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
      }, 400);
      return () => clearInterval(interval);
    }, []);
    return <span style={styles.thinking}>Thinking{dots}</span>;
  };

  return (
    <div style={styles.container}>
      <div ref={chatAreaRef} style={styles.chatArea}>
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.messageRow,
              ...(message.role === "user" ? styles.userRow : {}),
            }}
          >
            {message.role === "assistant" && (
              <div style={{ ...styles.avatar, ...styles.assistantAvatar }}>A</div>
            )}
            <div
              style={{
                ...styles.bubble,
                ...(message.role === "user" ? styles.userBubble : styles.assistantBubble),
              }}
            >
              {message.role === "assistant" && !message.content ? (
                <ThinkingIndicator />
              ) : (
                message.content
              )}
            </div>
            {message.role === "user" && (
              <div style={{ ...styles.avatar, ...styles.userAvatar }}>U</div>
            )}
          </div>
        ))}
        {isLoading && messages[messages.length - 1]?.role === "user" && (
          <div style={styles.messageRow}>
            <div style={{ ...styles.avatar, ...styles.assistantAvatar }}>A</div>
            <div style={{ ...styles.bubble, ...styles.assistantBubble }}>
              <ThinkingIndicator />
            </div>
          </div>
        )}
      </div>
      <div style={styles.inputArea}>
        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about Autonomy..."
            style={styles.input}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            style={{
              ...styles.button,
              ...(isLoading || !input.trim() ? styles.buttonDisabled : {}),
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
            Send
          </button>
        </form>
      </div>
    </div>
  );
};
