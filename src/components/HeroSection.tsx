import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Heart } from "lucide-react";
import { Button } from "@/components/ui/button";

// Placeholder images - replace with your own in /public/hero-images
// const heroImages = [
//   "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=1920&q=80",
//   "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=1920&q=80",
//   "https://images.unsplash.com/photo-1529634806980-85c3dd6d34ac?w=1920&q=80",
//   "https://images.unsplash.com/photo-1537633468519-5764b5d52d99?w=1920&q=80",
// ];
const heroImages = [
  "/hero-images/20250616_182620.jpg",
  "/hero-images/20250726_194450.jpg",
  "/hero-images/20250727_181405.jpg",
  "/hero-images/20250802_123333.jpg",
  "/hero-images/20250816_191535.jpg",
  "/hero-images/20250904_171202.jpg",
  "/hero-images/20251105_135843.jpg",
  "/hero-images/20251105_183633.jpg",
  "/hero-images/20251107_195310.jpg",
  "/hero-images/IMG-20241207-WA0052.jpg",
  "/hero-images/IMG-20241208-WA0006.jpg",
  "/hero-images/IMG-20250104-WA0035.jpg",
  "/hero-images/IMG-20250104-WA0064.jpg",
  "/hero-images/IMG-20250104-WA0077.jpg",
  "/hero-images/IMG-20250104-WA0079.jpg",
  "/hero-images/IMG-20250126-WA0006.jpg",
  "/hero-images/IMG-20250126-WA0015.jpg",
  "/hero-images/IMG-20250126-WA0042.jpg",
  "/hero-images/IMG-20250328-WA0031.jpg",
  "/hero-images/IMG-20250328-WA0053.jpg",
  "/hero-images/IMG-20250328-WA0065.jpg",
  "/hero-images/IMG-20250328-WA0086.jpg",
  "/hero-images/IMG-20250330-WA0055.jpg",
  "/hero-images/IMG-20250330-WA0234.jpg",
  "/hero-images/IMG-20250425-WA0037.jpg",
  "/hero-images/IMG-20250425-WA0046.jpg",
  "/hero-images/IMG-20250426-WA0025.jpg",
  "/hero-images/IMG-20250426-WA0042.jpg",
  "/hero-images/IMG-20250907-WA0068.jpg",
  "/hero-images/IMG-20250907-WA0102.jpg",
  "/hero-images/IMG-20250907-WA0138.jpg",
  "/hero-images/IMG-20250907-WA0180.jpg",
  "/hero-images/IMG-20250907-WA0210.jpg",
  "/hero-images/IMG-20250907-WA0284.jpg",
  "/hero-images/IMG20241207070128.jpg",
  "/hero-images/IMG20241222174351.jpg",
  "/hero-images/IMG20241222174358.jpg",
  "/hero-images/IMG20250208195224.jpg",
  "/hero-images/IMG20250308132109.jpg",
  "/hero-images/IMG20250329201315.jpg",
  "/hero-images/IMG20250425120352.jpg",
  "/hero-images/IMG_20241207_174749_0389.jpg",
];
const HeroSection = () => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentImageIndex((prev) => (prev + 1) % heroImages.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const scrollToChat = () => {
    const chatSection = document.getElementById("memory-chat");
    chatSection?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative h-screen w-full overflow-hidden">
      {/* Background Slideshow */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentImageIndex}
          initial={{ opacity: 0, scale: 1.1 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.5, ease: "easeInOut" }}
          className="absolute inset-0"
        >
          <div
            className="h-full w-full bg-cover bg-center"
            style={{ backgroundImage: `url(${heroImages[currentImageIndex]})` }}
          />
        </motion.div>
      </AnimatePresence>

      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-charcoal/30 via-charcoal/40 to-charcoal/70" />

      {/* Content */}
      <div className="relative z-10 flex h-full flex-col items-center justify-center px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="space-y-6"
        >
          {/* Decorative Hearts */}
          <motion.div
            className="flex justify-center gap-4"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <Heart className="h-6 w-6 animate-pulse-soft fill-rose-gold-light text-rose-gold-light opacity-80" />
            <Heart className="h-8 w-8 animate-heart-beat fill-rose-gold text-rose-gold" />
            <Heart className="h-6 w-6 animate-pulse-soft fill-rose-gold-light text-rose-gold-light opacity-80" />
          </motion.div>

          {/* Main Title */}
          <motion.h1
            className="font-serif text-5xl font-semibold tracking-wide text-cream md:text-7xl lg:text-8xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.7 }}
          >
            One Year of Us
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            className="mx-auto max-w-md font-sans text-lg font-light tracking-wide text-cream/90 md:text-xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1 }}
          >
            A digital time capsule of our memories
          </motion.p>

          {/* Decorative Line */}
          <motion.div
            className="mx-auto h-px w-32 bg-gradient-to-r from-transparent via-rose-gold to-transparent"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1.2, delay: 1.2 }}
          />
        </motion.div>

        {/* Enter Button */}
        <motion.div
          className="absolute bottom-16 md:bottom-24"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.5 }}
        >
          <Button
            onClick={scrollToChat}
            variant="ghost"
            className="group flex flex-col items-center gap-2 border-none bg-transparent text-cream hover:bg-transparent hover:text-rose-gold-light"
          >
            <span className="font-serif text-lg tracking-widest">Enter Capsule</span>
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            >
              <ChevronDown className="h-6 w-6" />
            </motion.div>
          </Button>
        </motion.div>
      </div>

      {/* Image Indicators */}
      <div className="absolute bottom-8 left-1/2 z-10 flex -translate-x-1/2 gap-2">
        {heroImages.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentImageIndex(index)}
            className={`h-2 rounded-full transition-all duration-300 ${
              index === currentImageIndex
                ? "w-8 bg-rose-gold"
                : "w-2 bg-cream/50 hover:bg-cream/70"
            }`}
          />
        ))}
      </div>
    </section>
  );
};

export default HeroSection;
