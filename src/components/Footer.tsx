import { motion } from "framer-motion";
import { Heart } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border bg-card/50 py-8 md:py-12">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="flex flex-col items-center gap-4 text-center"
        >
          <div className="flex items-center gap-2">
            <Heart className="h-5 w-5 animate-heart-beat fill-primary text-primary" />
          </div>

          <p className="font-serif text-lg text-foreground">
            Made with endless love
          </p>

          <p className="font-sans text-sm text-muted-foreground">
            Our Time Capsule • {currentYear}
          </p>

          <div className="h-px w-16 bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

          <p className="max-w-md font-serif text-sm italic text-muted-foreground">
            "And in the end, the love you take is equal to the love you make."
          </p>
        </motion.div>
      </div>
    </footer>
  );
};

export default Footer;
