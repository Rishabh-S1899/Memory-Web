import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Heart, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ChatMessage {
  id: string;
  role: "user" | "bot";
  text: string;
  images?: string[];
  timestamp: Date;
}

const TypingIndicator = () => (
  <div className="flex items-center gap-1.5 px-4 py-3">
    {[0, 1, 2].map((i) => (
      <motion.div
        key={i}
        className="text-rose-gold"
        style={{ animationDelay: `${i * 0.15}s` }}
        animate={{ y: [0, -8, 0] }}
        transition={{
          duration: 0.6,
          repeat: Infinity,
          delay: i * 0.15,
          ease: "easeInOut",
        }}
      >
        <Heart className="h-4 w-4 fill-rose-gold" />
      </motion.div>
    ))}
  </div>
);

const ChatBubble = ({ message }: { message: ChatMessage }) => {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[80%] space-y-3 rounded-2xl px-4 py-3 shadow-card ${
          isUser
            ? "rounded-br-sm bg-primary text-primary-foreground"
            : "rounded-bl-sm bg-card text-card-foreground border border-border"
        }`}
      >
        <p className="text-sm leading-relaxed md:text-base">{message.text}</p>

        {/* Rich Media Images */}
        {message.images && message.images.length > 0 && (
          <div className="space-y-2 pt-2">
            {message.images.map((img, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="overflow-hidden rounded-xl"
              >
                <img
                  src={img}
                  alt="Memory"
                  className="w-full rounded-xl object-cover shadow-soft transition-transform duration-300 hover:scale-105"
                />
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};

const MemoryChatbot = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "bot",
      text: "Welcome to our memory capsule, my love 💕 Ask me about any of our special moments, and I'll help you relive them...",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      text: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: inputValue.trim() }),
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: data.text || "I couldn't find that memory, but I cherish every moment with you.",
        images: data.images,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      // Fallback response when API is not available
      const fallbackMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: "I'm having trouble connecting right now, but know that every memory with you is precious to me. Try asking about our first date, our favorite song, or any special moment we've shared together 💕",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <section
      id="memory-chat"
      className="min-h-screen bg-gradient-to-b from-background via-secondary/30 to-background py-16 md:py-24"
    >
      <div className="container mx-auto max-w-3xl px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-8 text-center md:mb-12"
        >
          <h2 className="font-serif text-3xl font-semibold text-foreground md:text-4xl lg:text-5xl">
            Our Memory Lane
          </h2>
          <p className="mt-3 font-sans text-muted-foreground">
            Ask about any moment we've shared together
          </p>
          <div className="mx-auto mt-4 h-px w-24 bg-gradient-to-r from-transparent via-primary to-transparent" />
        </motion.div>

        {/* Chat Container */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="overflow-hidden rounded-3xl border border-border bg-card/80 shadow-elevated backdrop-blur-sm"
        >
          {/* Chat Header */}
          <div className="border-b border-border bg-card px-6 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                <Heart className="h-5 w-5 fill-primary text-primary" />
              </div>
              <div>
                <h3 className="font-serif text-lg font-medium text-foreground">Memory Keeper</h3>
                <p className="text-xs text-muted-foreground">Always here to remember with you</p>
              </div>
            </div>
          </div>

          {/* Messages Area */}
          <div className="h-[400px] overflow-y-auto p-4 md:h-[500px] md:p-6">
            <div className="space-y-4">
              <AnimatePresence>
                {messages.map((message) => (
                  <ChatBubble key={message.id} message={message} />
                ))}
              </AnimatePresence>

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start"
                >
                  <div className="rounded-2xl rounded-bl-sm border border-border bg-card shadow-card">
                    <TypingIndicator />
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="border-t border-border bg-card/50 p-4">
            <div className="flex items-center gap-3">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about our memories..."
                className="flex-1 rounded-xl border-border bg-background py-6 font-sans placeholder:text-muted-foreground focus-visible:ring-primary"
                disabled={isLoading}
              />
              <Button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="h-12 w-12 rounded-xl bg-primary text-primary-foreground shadow-soft transition-all hover:bg-primary/90 hover:shadow-card disabled:opacity-50"
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default MemoryChatbot;
