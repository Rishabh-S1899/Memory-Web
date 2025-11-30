import { motion } from "framer-motion";
import { Eye, Heart } from "lucide-react";

const mockPoems = [
  {
    id: 1,
    title: "The Day We Met",
    excerpt: "In a crowded room, our eyes first met,\nA moment frozen in time, I'll never forget...",
    date: "January 2024",
    category: "Beginnings",
  },
  {
    id: 2,
    title: "Your Smile",
    excerpt: "Like sunshine breaking through the clouds,\nYour smile lights up my darkest days...",
    date: "March 2024",
    category: "Moments",
  },
  {
    id: 3,
    title: "Together Forever",
    excerpt: "Through every season, every storm we face,\nI'll be right here, in our sacred space...",
    date: "June 2024",
    category: "Promises",
  },
  {
    id: 4,
    title: "Midnight Thoughts",
    excerpt: "When the world sleeps and stars shine bright,\nI think of you through the quiet night...",
    date: "August 2024",
    category: "Reflections",
  },
  {
    id: 5,
    title: "Our Song",
    excerpt: "Every note reminds me of your grace,\nThe melody of love written on your face...",
    date: "September 2024",
    category: "Music",
  },
  {
    id: 6,
    title: "One Year",
    excerpt: "365 days of laughter and tears,\nA beautiful journey through all our years...",
    date: "December 2024",
    category: "Milestones",
  },
];

const categoryColors: Record<string, string> = {
  Beginnings: "bg-primary/20 text-primary",
  Moments: "bg-accent/20 text-accent",
  Promises: "bg-secondary/20 text-secondary",
  Reflections: "bg-primary/30 text-primary",
  Music: "bg-accent/30 text-accent",
  Milestones: "bg-secondary/30 text-secondary",
};

export const PoemGallery = () => {
  return (
    <section id="poems" className="min-h-screen py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-5xl md:text-6xl font-bold mb-4 animate-shimmer">
            Love Letters & Poems
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Words penned from the heart, capturing our journey together
          </p>
        </motion.div>

        {/* Masonry Grid */}
        <div className="columns-1 md:columns-2 lg:columns-3 gap-6 space-y-6">
          {mockPoems.map((poem, index) => (
            <motion.div
              key={poem.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="break-inside-avoid"
            >
              <motion.div
                whileHover={{ y: -8, scale: 1.02 }}
                className="glass rounded-3xl p-8 group cursor-pointer relative overflow-hidden"
              >
                {/* Hover gradient overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-accent/5 to-secondary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                <div className="relative z-10">
                  {/* Category Badge */}
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-xs font-medium mb-4 ${
                      categoryColors[poem.category]
                    }`}
                  >
                    {poem.category}
                  </span>

                  {/* Title */}
                  <h3 className="text-2xl font-bold mb-3 group-hover:text-primary transition-colors">
                    {poem.title}
                  </h3>

                  {/* Date */}
                  <p className="text-sm text-muted-foreground mb-4">{poem.date}</p>

                  {/* Excerpt */}
                  <p className="text-foreground/80 whitespace-pre-line font-serif italic mb-6 leading-relaxed">
                    {poem.excerpt}
                  </p>

                  {/* Action Buttons */}
                  <div className="flex gap-3">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="flex-1 px-4 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-xl transition-colors flex items-center justify-center gap-2 text-sm font-medium"
                    >
                      <Eye className="w-4 h-4" />
                      <span>Read Full</span>
                    </motion.button>

                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-4 py-2 bg-accent/10 hover:bg-accent/20 text-accent rounded-xl transition-colors flex items-center justify-center gap-2 text-sm font-medium"
                    >
                      <Heart className="w-4 h-4" />
                      <span>Visualize</span>
                    </motion.button>
                  </div>
                </div>

                {/* Decorative heart */}
                <motion.div
                  className="absolute -bottom-4 -right-4 text-primary/10"
                  animate={{ rotate: [0, 10, 0] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                >
                  <Heart className="w-24 h-24" fill="currentColor" />
                </motion.div>
              </motion.div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
