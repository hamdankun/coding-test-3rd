"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, FileText } from "lucide-react";
import { chatApi } from "@/lib/api";
import { formatCurrency } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: any[];
  metrics?: any;
  timestamp: Date;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string>();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Create conversation on mount
    chatApi.createConversation().then((conv) => {
      setConversationId(conv.conversation_id);
    });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await chatApi.query(input, undefined, conversationId);

      const assistantMessage: Message = {
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        metrics: response.metrics,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        role: "assistant",
        content: `Sorry, I encountered an error: ${
          error.response?.data?.detail || error.message
        }`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto h-[calc(100vh-12rem)]">
      <div className="mb-4">
        <h1 className="text-4xl font-bold mb-2">Fund Analysis Chat</h1>
        <p className="text-gray-600">
          Ask questions about fund performance, metrics, and transactions
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-md flex flex-col h-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <FileText className="w-16 h-16 mx-auto" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Start a conversation
              </h3>
              <p className="text-gray-600 mb-6">Try asking questions like:</p>
              <div className="space-y-2 max-w-md mx-auto">
                <SampleQuestion
                  question="What is the current DPI?"
                  onClick={() => setInput("What is the current DPI?")}
                />
                <SampleQuestion
                  question="Calculate the IRR for this fund"
                  onClick={() => setInput("Calculate the IRR for this fund")}
                />
                <SampleQuestion
                  question="What does Paid-In Capital mean?"
                  onClick={() => setInput("What does Paid-In Capital mean?")}
                />
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <MessageBubble key={index} message={message} />
          ))}

          {loading && (
            <div className="flex items-center space-x-2 text-gray-500">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t p-4">
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the fund..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <Send className="w-4 h-4" />
              <span>Send</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-3xl ${isUser ? "ml-12" : "mr-12"}`}>
        <div
          className={`rounded-lg p-4 ${
            isUser ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Metrics Display */}
        {message.metrics && (
          <div className="mt-3 bg-white border border-gray-200 rounded-lg p-4">
            <h4 className="font-semibold text-sm text-gray-700 mb-2">
              Calculated Metrics
            </h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(message.metrics).map(([key, value]) => {
                if (value === null || value === undefined) return null;

                let displayValue: string;
                if (typeof value === "number" && key.includes("irr")) {
                  displayValue = `${value.toFixed(2)}%`;
                } else if (typeof value === "number") {
                  displayValue = formatCurrency(value);
                } else {
                  displayValue = String(value);
                }

                return (
                  <div key={key} className="text-sm">
                    <span className="text-gray-600">{key.toUpperCase()}:</span>{" "}
                    <span className="font-semibold">{displayValue}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Sources Display */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <details className="bg-white border border-gray-200 rounded-lg">
              <summary className="px-4 py-2 cursor-pointer text-sm font-medium text-gray-700 hover:bg-gray-50">
                View Sources ({message.sources.length})
              </summary>
              <div className="px-4 py-3 space-y-2 border-t">
                {message.sources.slice(0, 3).map((source, idx) => (
                  <div key={idx} className="text-xs bg-gray-50 p-2 rounded">
                    <p className="text-gray-700 line-clamp-2">
                      {source.content}
                    </p>
                    {source.score && (
                      <p className="text-gray-500 mt-1">
                        Relevance: {(source.score * 100).toFixed(0)}%
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}

        <p className="text-xs text-gray-500 mt-2">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}

function SampleQuestion({
  question,
  onClick,
}: {
  question: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-4 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg text-sm text-gray-700 transition"
    >
      &quot;{question}&quot;
    </button>
  );
}
