export const DocsChatVoice = () => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hey! I am your interface to a team of agents that were just created, just for you, on Autonomy Computer. This is a live demo. We have access to dozens of pages of documentation and general information about Autonomy! We are part of a multi-tenant application to demonstrate that every user gets a sandboxed set of agents - each with its own unique identity, state, memory, and tools.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [visitorId] = useState(() => Math.random().toString(36).slice(2));
  const [conversationId] = useState(() => Math.random().toString(36).slice(2));
  const chatAreaRef = useRef(null);

  // Voice state
  const [voiceState, setVoiceState] = useState("idle"); // idle, connecting, listening, processing, speaking
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const workletNodeRef = useRef(null);
  const playbackContextRef = useRef(null);
  const nextPlayTimeRef = useRef(0);
  const scheduledSourcesRef = useRef([]);
  const isRecordingRef = useRef(false);
  const isSpeakingRef = useRef(false);
  const lastAudioPlayTimeRef = useRef(0);

  const DOCS_API_BASE = "https://a9eb812238f753132652ae09963a05e9-docs.cluster.autonomy.computer";
  const DOCS_WS_BASE = "wss://a9eb812238f753132652ae09963a05e9-docs.cluster.autonomy.computer";

  const AUDIO_WORKLET_CODE = `
    class PCMProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
      }
      process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input && input[0]) {
          const inputData = input[0];
          for (let i = 0; i < inputData.length; i++) {
            this.buffer[this.bufferIndex++] = inputData[i];
            if (this.bufferIndex >= this.bufferSize) {
              const pcm16 = new Int16Array(this.bufferSize);
              let maxLevel = 0;
              for (let j = 0; j < this.bufferSize; j++) {
                const s = Math.max(-1, Math.min(1, this.buffer[j]));
                pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                const absVal = Math.abs(this.buffer[j]);
                if (absVal > maxLevel) maxLevel = absVal;
              }
              this.port.postMessage({ pcm16: pcm16.buffer, maxLevel }, [pcm16.buffer]);
              this.buffer = new Float32Array(this.bufferSize);
              this.bufferIndex = 0;
            }
          }
        }
        return true;
      }
    }
    registerProcessor('pcm-processor', PCMProcessor);
  `;

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    return () => {
      stopVoiceSession();
    };
  }, []);

  const clearAudioQueue = useCallback(() => {
    if (scheduledSourcesRef.current.length === 0) return;
    scheduledSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch {}
    });
    scheduledSourcesRef.current = [];
    if (playbackContextRef.current) {
      nextPlayTimeRef.current = playbackContextRef.current.currentTime;
    }
  }, []);

  const playAudioChunk = useCallback(async (base64Audio) => {
    try {
      if (!playbackContextRef.current) {
        playbackContextRef.current = new AudioContext({ sampleRate: 24000 });
        nextPlayTimeRef.current = playbackContextRef.current.currentTime + 0.05;
      }
      if (scheduledSourcesRef.current.length === 0) {
        lastAudioPlayTimeRef.current = Date.now();
      }
      const audioBytes = Uint8Array.from(atob(base64Audio), (c) => c.charCodeAt(0));
      const pcm16 = new Int16Array(audioBytes.buffer);
      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768.0;
      }
      const audioBuffer = playbackContextRef.current.createBuffer(1, float32.length, 24000);
      audioBuffer.getChannelData(0).set(float32);
      const source = playbackContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(playbackContextRef.current.destination);
      source.onended = () => {
        scheduledSourcesRef.current = scheduledSourcesRef.current.filter((s) => s !== source);
      };
      scheduledSourcesRef.current.push(source);
      const currentTime = playbackContextRef.current.currentTime;
      if (nextPlayTimeRef.current < currentTime) {
        nextPlayTimeRef.current = currentTime + 0.03;
      }
      source.start(nextPlayTimeRef.current);
      nextPlayTimeRef.current += audioBuffer.duration;
    } catch (error) {
      console.error("Error playing audio:", error);
    }
  }, []);

  const handleServerMessage = useCallback(
    async (data) => {
      const eventType = data.type;
      switch (eventType) {
        case "connected":
          break;
        case "audio":
          isSpeakingRef.current = true;
          setVoiceState("speaking");
          await playAudioChunk(data.audio);
          break;
        case "transcript":
          if (data.role === "user") {
            setMessages((prev) => [...prev, { role: "user", content: data.text }]);
          } else {
            setMessages((prev) => [...prev, { role: "assistant", content: data.text }]);
          }
          break;
        case "speech_started":
          if (scheduledSourcesRef.current.length > 0) {
            clearAudioQueue();
          }
          isSpeakingRef.current = false;
          setVoiceState("listening");
          break;
        case "speech_stopped":
          setVoiceState("processing");
          break;
        case "response_complete":
          isSpeakingRef.current = false;
          setVoiceState("listening");
          break;
        case "error":
          const errorMsg = data.error;
          if (errorMsg && !errorMsg.includes("no active response")) {
            console.error("Voice error:", errorMsg);
            setVoiceState("idle");
          }
          break;
      }
    },
    [clearAudioQueue, playAudioChunk],
  );

  const startVoiceSession = useCallback(async () => {
    try {
      setVoiceState("connecting");
      const wsUrl = `${DOCS_WS_BASE}/agents/docs/voice?scope=${visitorId}&conversation=${conversationId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "config" }));
      };
      ws.onmessage = async (event) => {
        try {
          const data = JSON.parse(event.data);
          await handleServerMessage(data);
        } catch (err) {
          console.error("Error handling message:", err);
        }
      };
      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setVoiceState("idle");
      };
      ws.onclose = () => {
        isRecordingRef.current = false;
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        if (workletNodeRef.current) {
          workletNodeRef.current.disconnect();
          workletNodeRef.current = null;
        }
        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        setVoiceState("idle");
      };

      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("Connection timeout")), 5000);
        ws.addEventListener("open", () => {
          clearTimeout(timeout);
          resolve();
        });
        ws.addEventListener("error", () => {
          clearTimeout(timeout);
          reject(new Error("Connection failed"));
        });
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 24000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 24000 });
      audioContextRef.current = audioContext;

      const blob = new Blob([AUDIO_WORKLET_CODE], { type: "application/javascript" });
      const workletUrl = URL.createObjectURL(blob);
      try {
        await audioContext.audioWorklet.addModule(workletUrl);
      } finally {
        URL.revokeObjectURL(workletUrl);
      }

      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (e) => {
        if (!isRecordingRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const { pcm16, maxLevel } = e.data;
        const timeSinceAudioStart = Date.now() - lastAudioPlayTimeRef.current;
        if (
          isSpeakingRef.current &&
          scheduledSourcesRef.current.length > 0 &&
          maxLevel > 0.25 &&
          timeSinceAudioStart > 1000
        ) {
          clearAudioQueue();
          isSpeakingRef.current = false;
          setVoiceState("listening");
        }
        const audioBase64 = btoa(String.fromCharCode(...new Uint8Array(pcm16)));
        wsRef.current.send(JSON.stringify({ type: "audio", audio: audioBase64 }));
      };

      source.connect(workletNode);
      workletNode.connect(audioContext.destination);
      isRecordingRef.current = true;
      setVoiceState("listening");
    } catch (error) {
      console.error("Error starting voice session:", error);
      setVoiceState("idle");
      stopVoiceSession();
    }
  }, [handleServerMessage, clearAudioQueue, visitorId, conversationId]);

  const stopRecording = useCallback(() => {
    isRecordingRef.current = false;
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  const stopVoiceSession = useCallback(() => {
    stopRecording();
    clearAudioQueue();
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "close" }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    if (playbackContextRef.current) {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    setVoiceState("idle");
  }, [stopRecording, clearAudioQueue]);

  const toggleVoice = useCallback(() => {
    if (voiceState === "idle") {
      startVoiceSession();
    } else {
      stopVoiceSession();
    }
  }, [voiceState, startVoiceSession, stopVoiceSession]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading || voiceState !== "idle") return;

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
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const parsed = JSON.parse(line);
            // Handle conversation_snippet format from Autonomy
            if (parsed.type === "conversation_snippet" && parsed.messages) {
              for (const msg of parsed.messages) {
                if (msg.role === "assistant" && msg.content?.text) {
                  fullText += msg.content.text;
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
              }
            }
            // Also handle simple text format as fallback
            else if (parsed.text) {
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
          } catch {}
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

  const getVoiceStatusText = () => {
    switch (voiceState) {
      case "connecting":
        return "Connecting...";
      case "listening":
        return "Listening...";
      case "processing":
        return "Thinking...";
      case "speaking":
        return "Speaking...";
      default:
        return "Click to talk";
    }
  };

  const ThinkingIndicator = () => {
    const [dots, setDots] = useState("");
    useEffect(() => {
      const interval = setInterval(() => {
        setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
      }, 400);
      return () => clearInterval(interval);
    }, []);
    return <span style={{ color: "var(--text-muted, #9ca3af)" }}>Thinking{dots}</span>;
  };

  const styles = {
    container: {
      border: "1px solid var(--border-color, #e5e7eb)",
      borderRadius: "12px",
      overflow: "hidden",
      backgroundColor: "var(--background-color, #fff)",
    },
    layout: {
      display: "flex",
      flexDirection: "row",
      height: "500px",
    },
    chatPanel: {
      flex: "1 1 70%",
      display: "flex",
      flexDirection: "column",
      borderRight: "1px solid var(--border-color, #e5e7eb)",
    },
    voicePanel: {
      flex: "1 1 30%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "24px",
      backgroundColor: "var(--background-color, #fff)",
      minWidth: "200px",
    },
    chatArea: {
      flex: 1,
      overflowY: "auto",
      padding: "16px",
      display: "flex",
      flexDirection: "column",
      gap: "12px",
      backgroundColor: "var(--background-muted, #f9fafb)",
      maxHeight: "400px",
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
      border: "2px solid transparent",
      fontSize: "14px",
      backgroundColor: "var(--background-color, #fff)",
      color: "var(--text-color, #374151)",
      outline: "none",
      backgroundImage:
        "linear-gradient(var(--background-color, #fff), var(--background-color, #fff)), linear-gradient(90deg, #f54a00, #ff8055, #f54a00)",
      backgroundOrigin: "border-box",
      backgroundClip: "padding-box, border-box",
      animation: "border-glow 3s linear infinite",
      backgroundSize: "100% 100%, 200% 100%",
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
      transition: "all 0.3s ease",
      animation: "button-pulse 2s ease-in-out infinite",
      boxShadow: "0 2px 10px rgba(245, 74, 0, 0.3)",
    },
    buttonDisabled: {
      opacity: 0.5,
      cursor: "not-allowed",
      animation: "none",
      boxShadow: "none",
    },
    voiceCircleContainer: {
      position: "relative",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    pulseRing: {
      position: "absolute",
      width: "100px",
      height: "100px",
      borderRadius: "50%",
      border: "2px solid rgba(245, 74, 0, 0.4)",
      animation: "pulse-ring 2.5s ease-out infinite",
    },
    voiceCircle: {
      position: "relative",
      width: "100px",
      height: "100px",
      borderRadius: "50%",
      background:
        voiceState === "idle"
          ? "linear-gradient(135deg, #f54a00 0%, #c44d24 100%)"
          : voiceState === "speaking"
            ? "linear-gradient(135deg, #ff8055 0%, #f54a00 100%)"
            : "linear-gradient(135deg, #f54a00 0%, #c44d24 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      cursor: isLoading ? "not-allowed" : "pointer",
      transition: "all 0.3s ease",
      boxShadow:
        voiceState === "idle"
          ? "0 10px 40px rgba(245, 74, 0, 0.3), inset 0 2px 10px rgba(255, 255, 255, 0.1)"
          : "0 10px 60px rgba(245, 74, 0, 0.5), 0 0 60px rgba(245, 74, 0, 0.35), inset 0 2px 15px rgba(255, 255, 255, 0.15)",
      opacity: isLoading ? 0.5 : 1,
      border: "none",
      animation: voiceState === "idle" ? "pulse-glow 2.5s ease-in-out infinite" : "none",
    },
    voiceStatus: {
      marginTop: "16px",
      fontSize: "14px",
      fontWeight: "500",
      color: voiceState !== "idle" ? "#f54a00" : "var(--text-muted, #9ca3af)",
    },
    voiceHint: {
      marginTop: "8px",
      fontSize: "12px",
      color: "var(--text-muted, #9ca3af)",
      textAlign: "center",
      maxWidth: "160px",
    },
    waveformBars: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "4px",
    },
    waveformBar: {
      width: "4px",
      backgroundColor: "white",
      borderRadius: "2px",
      transition: "height 0.1s ease",
    },
  };

  const WaveformIcon = ({ animated }) => {
    const heights = animated ? [16, 28, 22, 32, 18] : [12, 12, 12, 12, 12];
    return (
      <div style={styles.waveformBars}>
        {heights.map((h, i) => (
          <div
            key={i}
            style={{
              ...styles.waveformBar,
              height: `${h}px`,
              animation: animated ? `waveform 0.8s ease-in-out infinite` : "none",
              animationDelay: animated ? `${i * 0.1}s` : "0s",
            }}
          />
        ))}
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <style>{`
        @keyframes waveform {
          0%, 100% { transform: scaleY(1); }
          50% { transform: scaleY(1.5); }
        }
        @keyframes pulse-ring {
          0% {
            transform: scale(1);
            opacity: 0.8;
          }
          100% {
            transform: scale(1.5);
            opacity: 0;
          }
        }
        @keyframes pulse-glow {
          0%, 100% {
            box-shadow: 0 0 20px rgba(245, 74, 0, 0.4), 0 0 40px rgba(245, 74, 0, 0.2);
          }
          50% {
            box-shadow: 0 0 30px rgba(245, 74, 0, 0.6), 0 0 60px rgba(245, 74, 0, 0.3);
          }
        }
        @keyframes button-pulse {
          0%, 100% {
            box-shadow: 0 2px 10px rgba(245, 74, 0, 0.3);
            transform: scale(1);
          }
          50% {
            box-shadow: 0 4px 20px rgba(245, 74, 0, 0.5);
            transform: scale(1.02);
          }
        }
        @keyframes border-glow {
          0% {
            background-position: 0% 50%, 0% 50%;
          }
          100% {
            background-position: 0% 50%, 200% 50%;
          }
        }
        @media (max-width: 768px) {
          .docs-chat-layout { flex-direction: column !important; }
          .docs-chat-panel { border-right: none !important; border-bottom: 1px solid var(--border-color, #e5e7eb) !important; }
          .docs-voice-panel { min-height: 200px !important; padding: 16px !important; }
        }
      `}</style>
      <div className="docs-chat-layout" style={styles.layout}>
        <div className="docs-chat-panel" style={styles.chatPanel}>
          <div ref={chatAreaRef} style={styles.chatArea}>
            {messages.map((message, index) => (
              <div
                key={index}
                style={{
                  ...styles.messageRow,
                  ...(message.role === "user" ? styles.userRow : {}),
                }}
              >
                {message.role === "assistant" && <div style={{ ...styles.avatar, ...styles.assistantAvatar }}>A</div>}
                <div
                  style={{
                    ...styles.bubble,
                    ...(message.role === "user" ? styles.userBubble : styles.assistantBubble),
                  }}
                >
                  {message.role === "assistant" && !message.content ? <ThinkingIndicator /> : message.content}
                </div>
                {message.role === "user" && <div style={{ ...styles.avatar, ...styles.userAvatar }}>U</div>}
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
                disabled={isLoading || voiceState !== "idle"}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim() || voiceState !== "idle"}
                style={{
                  ...styles.button,
                  ...(isLoading || !input.trim() || voiceState !== "idle" ? styles.buttonDisabled : {}),
                }}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
                Send
              </button>
            </form>
          </div>
        </div>
        <div className="docs-voice-panel" style={styles.voicePanel}>
          <div style={styles.voiceCircleContainer}>
            {voiceState === "idle" && (
              <>
                <div style={{ ...styles.pulseRing }} />
                <div style={{ ...styles.pulseRing, animationDelay: "0.8s" }} />
                <div style={{ ...styles.pulseRing, animationDelay: "1.6s" }} />
              </>
            )}
            {(voiceState === "listening" || voiceState === "speaking" || voiceState === "processing") && (
              <>
                <div style={{ ...styles.pulseRing, animation: "pulse-ring 2s ease-out infinite" }} />
                <div
                  style={{ ...styles.pulseRing, animation: "pulse-ring 2s ease-out infinite", animationDelay: "0.5s" }}
                />
                <div
                  style={{ ...styles.pulseRing, animation: "pulse-ring 2s ease-out infinite", animationDelay: "1s" }}
                />
              </>
            )}
            <button
              onClick={toggleVoice}
              disabled={isLoading}
              style={styles.voiceCircle}
              aria-label={voiceState === "idle" ? "Start voice chat" : "Stop voice chat"}
            >
              <WaveformIcon animated={voiceState === "listening" || voiceState === "speaking"} />
            </button>
          </div>
          <div style={styles.voiceStatus}>{getVoiceStatusText()}</div>
          <div style={styles.voiceHint}>Click to start a voice conversation about Autonomy</div>
        </div>
      </div>
    </div>
  );
};
