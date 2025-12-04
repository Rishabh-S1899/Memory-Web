import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wand2, Download, ChevronDown, ChevronUp, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Poem, artStyles, ArtStyle } from "@/data/mockData";

interface PoemCardProps {
  poem: Poem;
}

const PoemCard = ({ poem }: PoemCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState<ArtStyle>("cinematic");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [showImage, setShowImage] = useState(false);

  const handleVisualize = async () => {
    setIsGenerating(true);

    try {
      const response = await fetch("http://localhost:8000/generate-wallpaper", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: poem.fullText,
          style: selectedStyle,
        }),
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();
      setGeneratedImage(data.image_url || data.imageUrl);
      setShowImage(true);
    } catch (error) {
      // Fallback placeholder image
      const fallbackImages: Record<ArtStyle, string> = {
        cinematic: "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=800&q=80",
        watercolor: "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=800&q=80",
        anime: "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=800&q=80",
        "oil-painting": "https://images.unsplash.com/photo-1579541513287-3f17a5d8d62c?w=800&q=80",
      };
      setGeneratedImage(fallbackImages[selectedStyle]);
      setShowImage(true);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (generatedImage) {
      const link = document.createElement("a");
      link.href = generatedImage;
      link.download = `${poem.title.replace(/\s+/g, "-").toLowerCase()}-${selectedStyle}.jpg`;
      link.target = "_blank";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const closeImage = () => {
    setShowImage(false);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="group relative"
    >
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-card transition-all duration-300 hover:shadow-elevated">
        {/* Card Content */}
        <div className="p-6 md:p-8">
          {/* Header */}
          <div className="mb-4">
            {poem.date && (
              <span className="mb-2 inline-block font-sans text-xs font-medium uppercase tracking-widest text-primary">
                {poem.date}
              </span>
            )}
            <h3 className="font-serif text-2xl font-semibold text-foreground md:text-3xl">
              {poem.title}
            </h3>
          </div>

          {/* Poem Text */}
          <AnimatePresence mode="wait">
            {isExpanded ? (
              <motion.div
                key="full"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="mb-6"
              >
                <p className="whitespace-pre-line font-sans text-base leading-relaxed text-muted-foreground">
                  {poem.fullText}
                </p>
              </motion.div>
            ) : (
              <motion.p
                key="preview"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mb-6 font-sans text-base italic text-muted-foreground"
              >
                "{poem.preview}"
              </motion.p>
            )}
          </AnimatePresence>

          {/* Controls */}
          <div className="space-y-4">
            {/* Expand/Collapse Button */}
            <Button
              variant="ghost"
              onClick={() => setIsExpanded(!isExpanded)}
              className="w-full justify-center gap-2 border border-border text-foreground hover:bg-secondary hover:text-foreground"
            >
              {isExpanded ? (
                <>
                  <ChevronUp className="h-4 w-4" />
                  Show Less
                </>
              ) : (
                <>
                  <ChevronDown className="h-4 w-4" />
                  Read Full Poem
                </>
              )}
            </Button>

            {/* Style Selector & Visualize */}
            <div className="flex flex-col gap-3 sm:flex-row">
              <Select
                value={selectedStyle}
                onValueChange={(value) => setSelectedStyle(value as ArtStyle)}
              >
                <SelectTrigger className="flex-1 border-border bg-background">
                  <SelectValue placeholder="Art Style" />
                </SelectTrigger>
                <SelectContent className="border-border bg-card">
                  {artStyles.map((style) => (
                    <SelectItem
                      key={style.value}
                      value={style.value}
                      className="focus:bg-secondary"
                    >
                      {style.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button
                onClick={handleVisualize}
                disabled={isGenerating}
                className="flex-1 gap-2 bg-primary text-primary-foreground shadow-soft hover:bg-primary/90 sm:flex-none"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Painting...
                  </>
                ) : (
                  <>
                    <Wand2 className="h-4 w-4" />
                    Visualize
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Generated Image Overlay */}
        <AnimatePresence>
          {showImage && generatedImage && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              className="absolute inset-0 z-10 flex flex-col overflow-hidden rounded-2xl bg-charcoal"
            >
              {/* Image */}
              <div className="relative flex-1">
                <img
                  src={generatedImage}
                  alt={`${poem.title} - ${selectedStyle}`}
                  className="h-full w-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-charcoal/80 via-transparent to-charcoal/30" />
              </div>

              {/* Overlay Content */}
              <div className="absolute bottom-0 left-0 right-0 p-6">
                <div className="mb-4 flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-rose-gold-light" />
                  <span className="font-sans text-xs uppercase tracking-widest text-cream/80">
                    {selectedStyle} Style
                  </span>
                </div>
                <h4 className="mb-4 font-serif text-2xl font-semibold text-cream">
                  {poem.title}
                </h4>

                <div className="flex gap-3">
                  <Button
                    onClick={handleDownload}
                    className="flex-1 gap-2 bg-rose-gold text-cream hover:bg-rose-gold/90"
                  >
                    <Download className="h-4 w-4" />
                    Download
                  </Button>
                  <Button
                    onClick={closeImage}
                    variant="outline"
                    className="flex-1 border-cream/30 bg-transparent text-cream hover:bg-cream/10 hover:text-cream"
                  >
                    Back to Poem
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default PoemCard;
