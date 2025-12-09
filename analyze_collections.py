import json
import os
from transformers import AutoTokenizer

# Configuration
FILES_TO_ANALYZE = [
    "parsed_books.jsonl",
    "parsed_papers.jsonl",
    "parsed_technical_manager.jsonl"
]
# Using a standard Qwen tokenizer as fallback/default if the specific one is not available
# Since Qwen3-30b likely uses the same tokenizer as Qwen 2.5 series.
TOKENIZER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507" 

def analyze_file(filepath, tokenizer):
    print(f"Analyzing {filepath}...")
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return

    total_tokens = 0
    doc_count = 0
    max_tokens = 0
    min_tokens = float('inf')
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    # Extract text. The previous dump script dumps the whole doc.
                    # We assume there's a 'text' field or we might need to look for content.
                    # Based on web_data_dump.py (line 46), 'text' is the field.
                    # If it's pure raw dump from mongo, it might differ.
                    # We will inspect the keys if 'text' is missing.
                    
                    text = data.get('full_text', '')
                    if not text:
                        text = data.get('text', '')
                    if not text:
                        text = data.get('content', '')
                    if not text:
                        text = data.get('abstract', '')
                    
                    if not text:
                        continue

                    tokens = tokenizer.encode(text)
                    count = len(tokens)
                    
                    total_tokens += count
                    doc_count += 1
                    if count > max_tokens:
                        max_tokens = count
                    if count < min_tokens:
                        min_tokens = count
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"  Error reading file: {e}")
        return

    if doc_count > 0:
        avg_tokens = total_tokens / doc_count
        print(f"  Documents: {doc_count}")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Avg Tokens: {avg_tokens:.2f}")
        print(f"  Max Tokens: {max_tokens}")
        print(f"  Min Tokens: {min_tokens}")
    else:
        print("  No valid documents found.")
    print("-" * 40)

def main():
    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer {TOKENIZER_ID}: {e}")
        print("Trying 'Qwen/Qwen2.5-7B-Instruct'...") # Fallback to a smaller one, same vocab
        try:
             tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        except Exception as e2:
            print(f"Failed to load fallback tokenizer: {e2}")
            return

    for filename in FILES_TO_ANALYZE:
        analyze_file(filename, tokenizer)

if __name__ == "__main__":
    main()
