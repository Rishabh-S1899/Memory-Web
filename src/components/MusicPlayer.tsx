import { useState } from "react";
import { motion } from "framer-motion";
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react";
import albumPlaceholder from "@/assets/album-placeholder.jpg";

// Mock data - ready for Spotify SDK integration
const mockTrack = {
  title: "Your Song",
  artist: "Our Favorite Artist",
  albumArt: albumPlaceholder,
  duration: 240000,
};

export const MusicPlayer = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(35);

  // Mock visualizer bars
  const visualizerBars = Array.from({ length: 40 }, (_, i) => ({
    id: i,
    height: Math.random() * 60 + 20,
  }));

  return (
    <motion.div
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ delay: 1.5, duration: 0.8 }}
      className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 w-full max-w-2xl px-6"
    >
      <div className="glass rounded-3xl p-6 shadow-2xl">
        <div className="flex items-center gap-6">
          {/* Album Art */}
          <motion.div
            className="relative w-20 h-20 rounded-2xl overflow-hidden flex-shrink-0"
            whileHover={{ scale: 1.05 }}
          >
            <img
              src={mockTrack.albumArt}
              alt="Album Art"
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
          </motion.div>

          {/* Track Info & Controls */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-3">
              <div className="min-w-0 flex-1 mr-4">
                <h4 className="font-semibold text-foreground truncate">
                  {mockTrack.title}
                </h4>
                <p className="text-sm text-muted-foreground truncate">
                  {mockTrack.artist}
                </p>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-2">
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 hover:bg-primary/10 rounded-full transition-colors"
                >
                  <SkipBack className="w-5 h-5 text-foreground/70" />
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-3 bg-primary rounded-full hover:bg-primary/90 transition-colors"
                >
                  {isPlaying ? (
                    <Pause className="w-5 h-5 text-primary-foreground" fill="currentColor" />
                  ) : (
                    <Play className="w-5 h-5 text-primary-foreground" fill="currentColor" />
                  )}
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 hover:bg-primary/10 rounded-full transition-colors"
                >
                  <SkipForward className="w-5 h-5 text-foreground/70" />
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 hover:bg-primary/10 rounded-full transition-colors ml-2"
                >
                  <Volume2 className="w-5 h-5 text-foreground/70" />
                </motion.button>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="relative h-1 bg-muted rounded-full overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 bg-primary"
                style={{ width: `${progress}%` }}
                animate={{ width: isPlaying ? `${progress + 1}%` : `${progress}%` }}
                transition={{ duration: 1 }}
              />
            </div>
          </div>
        </div>

        {/* Visualizer */}
        <div className="flex items-end justify-center gap-1 h-16 mt-4">
          {visualizerBars.map((bar) => (
            <motion.div
              key={bar.id}
              className="w-1 bg-gradient-to-t from-primary via-accent to-secondary rounded-full"
              initial={{ height: 4 }}
              animate={{
                height: isPlaying ? [4, bar.height, 4] : 4,
              }}
              transition={{
                duration: 0.6,
                repeat: Infinity,
                ease: "easeInOut",
                delay: bar.id * 0.03,
              }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
};
