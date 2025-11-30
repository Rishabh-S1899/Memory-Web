import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import mapboxgl from "mapbox-gl";
import { Heart, X } from "lucide-react";
import "mapbox-gl/dist/mapbox-gl.css";

// Mock data - will connect to API
const mockMemories = [
  {
    id: 1,
    lat: 28.6139,
    lng: 77.2090,
    title: "First Date",
    description: "Where it all began... that magical evening when I first saw your smile.",
    image: "https://images.unsplash.com/photo-1551218808-94e220e084d2?w=400",
    date: "January 15, 2024",
  },
  {
    id: 2,
    lat: 28.5355,
    lng: 77.3910,
    title: "The Coffee Shop",
    description: "Our favorite spot where we spent hours just talking and laughing.",
    image: "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400",
    date: "March 8, 2024",
  },
  {
    id: 3,
    lat: 28.7041,
    lng: 77.1025,
    title: "Weekend Getaway",
    description: "The trip that made us realize we're meant to be together forever.",
    image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
    date: "June 22, 2024",
  },
  {
    id: 4,
    lat: 28.4595,
    lng: 77.0266,
    title: "Starlit Promise",
    description: "Under the stars, we promised each other forever and always.",
    image: "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400",
    date: "September 14, 2024",
  },
];

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || "";

export const MemoryMap = () => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [selectedMemory, setSelectedMemory] = useState<typeof mockMemories[0] | null>(null);
  const markersRef = useRef<mapboxgl.Marker[]>([]);

  useEffect(() => {
    if (!mapContainer.current || !MAPBOX_TOKEN) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/light-v11",
      center: [77.2090, 28.6139],
      zoom: 10,
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), "top-right");

    // Create custom heart markers
    mockMemories.forEach((memory) => {
      const el = document.createElement("div");
      el.className = "custom-marker";
      el.innerHTML = `
        <svg width="32" height="32" viewBox="0 0 24 24" fill="hsl(345, 75%, 70%)" stroke="white" stroke-width="2">
          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
        </svg>
      `;
      el.style.cursor = "pointer";
      el.style.filter = "drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2))";
      el.style.transition = "transform 0.3s ease";
      
      el.addEventListener("mouseenter", () => {
        el.style.transform = "scale(1.2)";
      });
      
      el.addEventListener("mouseleave", () => {
        el.style.transform = "scale(1)";
      });

      el.addEventListener("click", () => {
        setSelectedMemory(memory);
        map.current?.flyTo({
          center: [memory.lng, memory.lat],
          zoom: 12,
          duration: 1500,
        });
      });

      const marker = new mapboxgl.Marker(el)
        .setLngLat([memory.lng, memory.lat])
        .addTo(map.current!);
      
      markersRef.current.push(marker);
    });

    return () => {
      markersRef.current.forEach((marker) => marker.remove());
      map.current?.remove();
    };
  }, []);

  return (
    <section id="map" className="min-h-screen py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-5xl md:text-6xl font-bold mb-4 animate-shimmer">
            Our Memory Map
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Every pin marks a place where we created beautiful memories together
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="glass rounded-3xl overflow-hidden h-[600px] relative"
        >
          {MAPBOX_TOKEN ? (
            <>
              <div ref={mapContainer} className="w-full h-full" />

              <AnimatePresence>
                {selectedMemory && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.8, y: 20 }}
                    className="absolute top-6 left-6 right-6 md:left-auto md:w-96 glass-dark rounded-2xl p-6 text-white z-10"
                  >
                    <button
                      onClick={() => setSelectedMemory(null)}
                      className="absolute top-4 right-4 p-2 hover:bg-white/20 rounded-full transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>

                    <img
                      src={selectedMemory.image}
                      alt={selectedMemory.title}
                      className="w-full h-48 object-cover rounded-xl mb-4"
                    />

                    <h4 className="font-bold text-xl mb-2">{selectedMemory.title}</h4>
                    <p className="text-sm text-white/70 mb-3">{selectedMemory.date}</p>
                    <p className="text-sm text-white/90 leading-relaxed">
                      {selectedMemory.description}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md px-6">
                <Heart className="w-16 h-16 text-primary mx-auto mb-4" />
                <p className="text-lg text-muted-foreground mb-4">
                  Mapbox token required to display the interactive memory map
                </p>
                <p className="text-sm text-muted-foreground">
                  Add <code className="px-2 py-1 bg-muted rounded">VITE_MAPBOX_TOKEN</code> to your environment variables
                </p>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </section>
  );
};
