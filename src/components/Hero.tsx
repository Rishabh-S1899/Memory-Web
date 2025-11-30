import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Heart, Music, Map, Scroll } from "lucide-react";

// Import your slideshow images here
// Example:
// import image1 from "@/assets/slideshow/image1.jpg";
// import image2 from "@/assets/slideshow/image2.jpg";
// import image3 from "@/assets/slideshow/image3.jpg";

// Add all your imported images to this array
const slideshowImages: string[] = [
  // image1,
  // image2,
  // image3,
];

// Fallback image if no images provided
import heroBg from "@/assets/hero-bg.jpg";
const defaultImages = [heroBg];

interface HeroProps {
  onNavigate: (section: string) => void;
}

export const Hero = ({ onNavigate }: HeroProps) => {
  const images = slideshowImages.length > 0 ? slideshowImages : defaultImages;
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // Auto-advance slideshow every 8 seconds
  useEffect(() => {
    if (images.length <= 1) return;
    
    const interval = setInterval(() => {
      setCurrentImageIndex((prev) => (prev + 1) % images.length);
    }, 8000);

    return () => clearInterval(interval);
  }, [images.length]);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated background slideshow */}
      <div className="absolute inset-0 z-0">
        <AnimatePresence mode="wait">
          <motion.img
            key={currentImageIndex}
            src={images[currentImageIndex]}
            alt="Background"
            initial={{ opacity: 0, scale: 1 }}
            animate={{ opacity: 0.7, scale: 1.1 }}
            exit={{ opacity: 0, scale: 1.2 }}
            transition={{ duration: 2, ease: "easeInOut" }}
            className="absolute inset-0 w-full h-full object-cover"
          />
        </AnimatePresence>
        <div className="absolute inset-0 bg-gradient-to-b from-background/30 via-background/50 to-background/80" />
      </div>

      {/* Floating hearts */}
      <motion.div
        className="absolute top-20 left-20 text-primary/30"
        animate={{ y: [0, -20, 0], rotate: [0, 10, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      >
        <Heart className="w-12 h-12" fill="currentColor" />
      </motion.div>
      
      <motion.div
        className="absolute top-40 right-32 text-accent/30"
        animate={{ y: [0, -30, 0], rotate: [0, -15, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
      >
        <Heart className="w-8 h-8" fill="currentColor" />
      </motion.div>

      <motion.div
        className="absolute bottom-32 left-40 text-secondary/30"
        animate={{ y: [0, -25, 0], rotate: [0, 20, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 2 }}
      >
        <Heart className="w-10 h-10" fill="currentColor" />
      </motion.div>

      {/* Content */}
      <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
        >
          <h1 className="text-6xl md:text-8xl font-bold mb-6 animate-shimmer">
            Happy 1st Anniversary
          </h1>
          <motion.p
            className="text-2xl md:text-3xl text-foreground/80 mb-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 1 }}
          >
            My Dearest Love
          </motion.p>
          <motion.p
            className="text-lg md:text-xl text-muted-foreground mb-12 max-w-2xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8, duration: 1 }}
          >
            A year of memories, laughter, and love. This digital time capsule holds 
            all the moments that made our journey unforgettable.
          </motion.p>
        </motion.div>

        {/* Navigation buttons */}
        <motion.div
          className="flex flex-wrap gap-4 justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.8 }}
        >
          <button
            onClick={() => onNavigate("mood")}
            className="glass px-8 py-4 rounded-2xl hover:scale-105 transition-all duration-300 flex items-center gap-3 group"
          >
            <Music className="w-5 h-5 text-primary group-hover:animate-pulse" />
            <span className="font-medium">Our Soundtrack</span>
          </button>
          
          <button
            onClick={() => onNavigate("map")}
            className="glass px-8 py-4 rounded-2xl hover:scale-105 transition-all duration-300 flex items-center gap-3 group"
          >
            <Map className="w-5 h-5 text-accent group-hover:animate-pulse" />
            <span className="font-medium">Memory Map</span>
          </button>
          
          <button
            onClick={() => onNavigate("poems")}
            className="glass px-8 py-4 rounded-2xl hover:scale-105 transition-all duration-300 flex items-center gap-3 group"
          >
            <Scroll className="w-5 h-5 text-secondary group-hover:animate-pulse" />
            <span className="font-medium">Love Letters</span>
          </button>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          <div className="w-6 h-10 border-2 border-primary/50 rounded-full flex items-start justify-center p-2">
            <div className="w-1 h-2 bg-primary rounded-full animate-pulse" />
          </div>
        </motion.div>
      </div>
    </section>
  );
};
