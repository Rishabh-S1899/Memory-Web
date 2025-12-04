import os

# CONFIGURATION
# 1. Where are the images on your disk right now?
IMAGE_FOLDER_PATH = r"./public/hero-images" 

# 2. What should the path look like in React? 
# (e.g. if you put them in public/hero-images, this should be "/hero-images/")
REACT_PUBLIC_PREFIX = "/hero-images/"

def generate_list():
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_list = []

    try:
        # Get all files
        files = os.listdir(IMAGE_FOLDER_PATH)
        
        # Sort them so they appear in order (optional)
        files.sort()

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                # Create the formatted string
                full_path = f"{REACT_PUBLIC_PREFIX}{filename}"
                image_list.append(full_path)

        # Print the array ready for Copy-Paste
        print("Copy this array into your React code:\n")
        print("const heroImages = [")
        for img in image_list:
            print(f'  "{img}",')
        print("];")
        
        print(f"\n✅ Found {len(image_list)} images.")

    except FileNotFoundError:
        print(f"❌ Error: Could not find folder: {IMAGE_FOLDER_PATH}")

if __name__ == "__main__":
    generate_list()
