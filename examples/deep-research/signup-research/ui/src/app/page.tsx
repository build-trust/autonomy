"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Search, User, Building2, Mail, Loader2, ExternalLink } from "lucide-react";

// Simple markdown renderer component
function MarkdownRenderer({ content }: { content: string }) {
  const renderMarkdown = (text: string) => {
    const lines = text.split("\n");
    const elements: JSX.Element[] = [];
    let key = 0;

    for (const line of lines) {
      // H1
      if (line.startsWith("# ")) {
        elements.push(
          <h1 key={key++} className="text-2xl font-bold text-slate-900 mt-6 mb-3 first:mt-0">
            {line.slice(2)}
          </h1>,
        );
      }
      // H2
      else if (line.startsWith("## ")) {
        elements.push(
          <h2 key={key++} className="text-xl font-semibold text-slate-800 mt-5 mb-2 border-b border-slate-200 pb-1">
            {line.slice(3)}
          </h2>,
        );
      }
      // H3
      else if (line.startsWith("### ")) {
        elements.push(
          <h3 key={key++} className="text-lg font-medium text-slate-700 mt-4 mb-2">
            {line.slice(4)}
          </h3>,
        );
      }
      // Bold text handling within paragraphs
      else if (line.trim()) {
        // Replace **text** with bold spans
        const parts = line.split(/(\*\*[^*]+\*\*)/g);
        const formattedParts = parts.map((part, i) => {
          if (part.startsWith("**") && part.endsWith("**")) {
            return (
              <strong key={i} className="font-semibold text-slate-900">
                {part.slice(2, -2)}
              </strong>
            );
          }
          return part;
        });
        elements.push(
          <p key={key++} className="text-slate-600 leading-relaxed mb-3">
            {formattedParts}
          </p>,
        );
      }
      // Empty line
      else {
        elements.push(<div key={key++} className="h-2" />);
      }
    }

    return elements;
  };

  return <div className="space-y-1">{renderMarkdown(content)}</div>;
}

export default function Home() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [organization, setOrganization] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (resultRef.current) {
      resultRef.current.scrollTop = resultRef.current.scrollHeight;
    }
  }, [result]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setResult("");
    setError("");
    setStatus("Connecting...");

    try {
      const response = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, organization }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let isFirstContent = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            try {
              const parsed = JSON.parse(line);

              if (parsed.type === "error") {
                setError(parsed.message);
                continue;
              }

              if (parsed.messages) {
                for (const message of parsed.messages) {
                  // Skip tool results (they have tool_call_id)
                  if (message.tool_call_id) {
                    setStatus("Analyzing search results...");
                    continue;
                  }

                  // Skip messages that are just tool calls (no text content)
                  if (message.tool_calls && message.tool_calls.length > 0) {
                    // Check if this is a search tool call
                    const toolCall = message.tool_calls[0];
                    if (toolCall?.function?.name === "linkup_search") {
                      setStatus("Searching the web...");
                    } else if (toolCall?.function?.name === "linkup_fetch") {
                      setStatus("Fetching page content...");
                    }
                    continue;
                  }

                  // Only process assistant messages with actual text
                  if (message.role === "assistant" && message?.content?.text) {
                    const text = message.content.text;

                    // Skip empty text
                    if (!text.trim()) continue;

                    // Update status on first real content
                    if (isFirstContent && text.trim()) {
                      setStatus("Generating report...");
                      isFirstContent = false;
                    }

                    setResult((prev) => prev + text);
                  }
                }
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        try {
          const parsed = JSON.parse(buffer);
          if (parsed.messages) {
            for (const message of parsed.messages) {
              if (message.tool_call_id) continue;
              if (message.tool_calls && message.tool_calls.length > 0) continue;
              if (message.role === "assistant" && message?.content?.text) {
                setResult((prev) => prev + message.content.text);
              }
            }
          }
        } catch {
          // Skip malformed JSON
        }
      }

      setStatus("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setStatus("");
    } finally {
      setIsLoading(false);
    }
  };

  const clearForm = () => {
    setName("");
    setEmail("");
    setOrganization("");
    setResult("");
    setError("");
    setStatus("");
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-slate-50 to-indigo-50 p-4 md:p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2 py-6">
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Search className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Signup Research
            </h1>
          </div>
          <p className="text-lg text-slate-600">AI-powered intelligence for new user signups</p>
        </div>

        {/* Input Form */}
        <Card className="border-slate-200 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-t-lg border-b">
            <CardTitle className="flex items-center gap-2 text-slate-800">
              <User className="h-5 w-5 text-blue-600" />
              New User Information
            </CardTitle>
            <CardDescription>
              Enter the details of the new signup to research their background and company
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="name" className="flex items-center gap-2 text-slate-700">
                    <User className="h-4 w-4 text-blue-500" />
                    Name
                  </Label>
                  <Input
                    id="name"
                    placeholder="John Doe"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    disabled={isLoading}
                    className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email" className="flex items-center gap-2 text-slate-700">
                    <Mail className="h-4 w-4 text-blue-500" />
                    Email
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="john@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    disabled={isLoading}
                    className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="organization" className="flex items-center gap-2 text-slate-700">
                    <Building2 className="h-4 w-4 text-blue-500" />
                    Organization
                  </Label>
                  <Input
                    id="organization"
                    placeholder="Acme Inc"
                    value={organization}
                    onChange={(e) => setOrganization(e.target.value)}
                    required
                    disabled={isLoading}
                    className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
              </div>
              <div className="flex gap-3">
                <Button
                  type="submit"
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                  disabled={isLoading || !name || !email || !organization}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Researching...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-4 w-4" />
                      Research User
                    </>
                  )}
                </Button>
                {(result || error) && !isLoading && (
                  <Button type="button" variant="outline" onClick={clearForm} className="border-slate-300">
                    Clear
                  </Button>
                )}
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Status Indicator */}
        {isLoading && status && (
          <div className="flex items-center justify-center gap-3 py-4">
            <div className="relative">
              <div className="w-3 h-3 bg-blue-600 rounded-full animate-ping absolute"></div>
              <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
            </div>
            <span className="text-slate-600 font-medium">{status}</span>
          </div>
        )}

        {/* Results */}
        {(result || error) && (
          <Card className="border-slate-200 shadow-lg overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-green-50 to-emerald-50 border-b border-green-100">
              <CardTitle className="flex items-center gap-2 text-slate-800">
                <div className="p-1.5 bg-green-600 rounded">
                  <Search className="h-4 w-4 text-white" />
                </div>
                Research Results
                {isLoading && (
                  <span className="text-sm font-normal text-slate-500 ml-2 flex items-center gap-2">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Generating...
                  </span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {error && (
                <div className="bg-red-50 border-b border-red-200 text-red-700 px-6 py-4">
                  <strong>Error:</strong> {error}
                </div>
              )}
              <div ref={resultRef} className="p-6 min-h-[300px] max-h-[600px] overflow-y-auto bg-white">
                {result ? (
                  <MarkdownRenderer content={result} />
                ) : (
                  <span className="text-slate-400 italic">Research results will appear here...</span>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Footer */}
        <div className="text-center text-sm text-slate-500 py-4 flex items-center justify-center gap-1">
          Powered by{" "}
          <a
            href="https://linkup.so"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline inline-flex items-center gap-1 font-medium"
          >
            Linkup
            <ExternalLink className="h-3 w-3" />
          </a>{" "}
          web search API &{" "}
          <a
            href="https://autonomy.computer"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline inline-flex items-center gap-1 font-medium"
          >
            Autonomy
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>
    </main>
  );
}
