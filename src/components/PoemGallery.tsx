import { motion } from "framer-motion";
import { BookHeart } from "lucide-react";
import PoemCard from "./PoemCard";
import { poems } from "@/data/mockData";

const PoemGallery = () => {
  return (
    <section
      id="poem-gallery"
      className="min-h-screen bg-gradient-to-b from-background via-blush/20 to-background py-16 md:py-24"
    >
      <div className="container mx-auto max-w-6xl px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-12 text-center md:mb-16"
        >
          <motion.div
            initial={{ scale: 0 }}
            whileInView={{ scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10"
          >
            <BookHeart className="h-8 w-8 text-primary" />
          </motion.div>

          <h2 className="font-serif text-3xl font-semibold text-foreground md:text-4xl lg:text-5xl">
            Love Letters & Poems
          </h2>
          <p className="mx-auto mt-4 max-w-lg font-sans text-muted-foreground">
            Words written from the heart, waiting to be transformed into art. Select a poem
            and watch it come alive.
          </p>
          <div className="mx-auto mt-6 h-px w-24 bg-gradient-to-r from-transparent via-primary to-transparent" />
        </motion.div>

        {/* Poems Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:gap-8">
          {poems.map((poem, index) => (
            <motion.div
              key={poem.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <PoemCard poem={poem} />
            </motion.div>
          ))}
        </div>

        {/* Footer Note */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-16 text-center"
        >
          <p className="font-serif text-lg italic text-muted-foreground">
            "Every love story is beautiful, but ours is my favorite."
          </p>
        </motion.div>
      </div>
    </section>
  );
};

export default PoemGallery;
