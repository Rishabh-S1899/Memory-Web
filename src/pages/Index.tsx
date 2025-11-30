import { useRef } from "react";
import { Hero } from "@/components/Hero";
import { MusicPlayer } from "@/components/MusicPlayer";
import { MemoryMap } from "@/components/MemoryMap";
import { ChatWidget } from "@/components/ChatWidget";
import { PoemGallery } from "@/components/PoemGallery";

const Index = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const poemsRef = useRef<HTMLDivElement>(null);

  const handleNavigate = (section: string) => {
    const refs = {
      map: mapRef,
      poems: poemsRef,
      mood: null, // Mood board is on the hero
    };

    const targetRef = refs[section as keyof typeof refs];
    if (targetRef?.current) {
      targetRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="relative">
      {/* Hero Section */}
      <Hero onNavigate={handleNavigate} />

      {/* Memory Map Section */}
      <div ref={mapRef}>
        <MemoryMap />
      </div>

      {/* Poem Gallery Section */}
      <div ref={poemsRef}>
        <PoemGallery />
      </div>

      {/* Music Player - Fixed at bottom */}
      <MusicPlayer />

      {/* Chat Widget - Fixed at bottom right */}
      <ChatWidget />
    </div>
  );
};

export default Index;
