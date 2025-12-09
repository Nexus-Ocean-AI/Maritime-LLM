import json
import os
import glob

# Configuration
INPUT_DIR = "data"
OUTPUT_FILE = "data/maritime_pretraining_data.jsonl"
MIN_TEXT_LENGTH = 50  # Filter out very short texts

def get_text_content(doc):
    """
    Extract text content from document using various possible keys.
    Returns the found text or None.
    """
    # Priority: full_text -> text -> content
    # Common in parsed_books, parsed_papers, parsed_technical_manager
    if "full_text" in doc and doc["full_text"]:
        return doc["full_text"]
    
    # Common in web scrapes
    if "text" in doc and doc["text"]:
        return doc["text"]
        
    if "content" in doc and doc["content"]:
        return doc["content"]
    
    # Fallback for papers if no full text is parsing
    if "abstract" in doc and doc["abstract"]:
        # Maybe combine title and abstract?
        parts = []
        if doc.get("title"):
            parts.append(doc["title"])
        parts.append(doc["abstract"])
        return "\n\n".join(parts)

    return None

def main():
    # Ensure output directory exists (files are expected in 'data' anyway)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    input_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    # Avoid reading the output file if it's in the same list
    input_files = [f for f in input_files if os.path.abspath(f) != os.path.abspath(OUTPUT_FILE)]
    
    print(f"Found {len(input_files)} input files: {input_files}")
    
    total_docs = 0
    total_chars = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for file_path in input_files:
            print(f"Processing {file_path}...")
            file_docs = 0
            
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line_num, line in enumerate(in_f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            text = get_text_content(data)
                            
                            if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                                # Standard Format for Pre-training: {"text": "..."}
                                # We can also keep metadata if needed, but usually 'text' is what matters.
                                output_entry = {
                                    "text": text,
                                    # Optional: keep track of source for data mixture handling later
                                    "meta": {
                                        "source_file": os.path.basename(file_path),
                                        "id": str(data.get("_id", ""))
                                    }
                                }
                                
                                out_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                                file_docs += 1
                                total_chars += len(text)
                                total_docs += 1
                                
                        except json.JSONDecodeError:
                            print(f"  [Warning] JSON Decode Error at line {line_num} in {file_path}")
                            continue
                            
            except Exception as e:
                print(f"  [Error] Failed to process {file_path}: {e}")
            
            print(f"  -> Added {file_docs} documents.")

    print("="*40)
    print(f"Processing Complete.")
    print(f"Total Documents: {total_docs}")
    print(f"Total Characters: {total_chars}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()
