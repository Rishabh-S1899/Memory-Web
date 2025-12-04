import Navigation from "@/components/Navigation";
import HeroSection from "@/components/HeroSection";
import MemoryChatbot from "@/components/MemoryChatbot";
import PoemGallery from "@/components/PoemGallery";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main>
        <HeroSection />
        <MemoryChatbot />
        <PoemGallery />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
