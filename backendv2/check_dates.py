import chromadb
import os

DB_PATH = "./chroma_db_data"

def main():
    print(f"🕵️‍♂️ Analyzing Date Metadata in {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        chat_col = client.get_collection("chat_memories")
        # Get ALL metadata (limit to 2000 to be safe)
        data = chat_col.get(include=['metadatas'])
        
        dates = []
        for meta in data['metadatas']:
            if 'date' in meta:
                dates.append(meta['date']) # Format YYYY-MM-DD
        
        # Sort and group by Month
        dates.sort()
        unique_months = sorted(list(set([d[:7] for d in dates]))) # Extract YYYY-MM
        
        print("\n✅ Found Data for these Months:")
        for m in unique_months:
            count = sum(1 for d in dates if d.startswith(m))
            print(f"   - {m}: {count} conversation chunks")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()