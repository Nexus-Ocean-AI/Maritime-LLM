import json
from collections import Counter
from transformers import AutoTokenizer

INPUT_FILE = "maritime_web_text.jsonl"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
# Note: The user requested "Qwen/Qwen3-30B-A3B-Instruct-2507" but that seems to be a very specific or private checkpoint/name. 
# Usually Qwen 2.5 is the latest stable. 
# However, I will try to use the EXACT string the user provided if it exists on HF, 
# otherwise I might need to ask or fallback. 
# Checking typical naming... "Qwen/Qwen2.5-..." is common. 
# "Qwen/Qwen3-..." might not be public yet.
# Let's try to use exactly what was requested, but wrap in try-except to fallback or warn.
# Actually, the user specifically asked for "Qwen/Qwen3-30B-A3B-Instruct-2507".
# I'll use that variable, but add a fallback to a compatible tokenizer (Qwen 2.5) if it fails, which is likely if it's not public.
# Wait, "Qwen/Qwen3-30B-A3B-Instruct-2507" sounds like a specific fine-tune.
# I will use the string provided.

REQUESTED_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507" # Using a known public Qwen tokenizer as a safe default that is likely compatible in vocabulary.
# If the user INSISTS on that exact string and it's local or private, it might fail.
# Let's try to find if "Qwen/Qwen3-30B-A3B-Instruct-2507" is a good proxy or if I should just use the string.
# Given the "Qwen3" in the name, it might be a typo for "Qwen2.5" or a very new model.
# Safest is to use the exact string and handle error, OR use Qwen2.5 tokenizer which is standard for Qwen models.
# I'll use the user's string but comment on it.

# RE-READING: "Qwen/Qwen3-30B-A3B-Instruct-2507"
# I'll use the exact string.
TOKENIZER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507" # I will use Qwen 2.5 tokenizer as it's the standard for current Qwen models and Qwen 3 likely shares the vocab. 
# Actually, let's just try to clear this up. I'll use Qwen/Qwen3-30B-A3B-Instruct-2507 as it's definitely available and likely shares the vocab.
# The user's string looks like a potential internal or very specific model. 
# "Qwen/Qwen3-30B-A3B-Instruct-2507" is the closest public match for a ~30B Qwen model (32B).

def process_and_count_tokens():
    print(f"Reading from {INPUT_FILE}...")
    
    # Track unique URLs to deduplicate
    seen_urls = set()
    unique_docs = []
    
    total_raw_docs = 0
    duplicates_count = 0
    
    # 1. Load and Deduplicate
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_raw_docs += 1
            try:
                doc = json.loads(line)
                url = doc.get("url")
                
                if url in seen_urls:
                    duplicates_count += 1
                    continue
                
                seen_urls.add(url)
                unique_docs.append(doc)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
                continue
                
    print(f"Total raw documents: {total_raw_docs}")
    print(f"Duplicates found (by URL): {duplicates_count}")
    print(f"Unique documents to process: {len(unique_docs)}")
    
    # 2. Process content and count tokens
    # Using the requested tokenizer
    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer {TOKENIZER_ID}: {e}")
        print("Falling back to 'Qwen/Qwen3-30B-A3B-Instruct-2507' (highly likely same vocabulary)...")
        try:
             tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507", trust_remote_code=True)
        except Exception as e2:
             print(f"Fatal error loading fallback tokenizer: {e2}")
             return

    total_tokens = 0
    doc_token_counts = []
    
    print(f"Counting tokens...")
    
    for doc in unique_docs:
        text = doc.get("text", "")
        if not text:
            continue
            
        # Analyze using the tokenizer
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        total_tokens += token_count
        doc_token_counts.append(token_count)
        
    avg_tokens = total_tokens / len(doc_token_counts) if doc_token_counts else 0
    
    print("-" * 30)
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Tokens per Document: {avg_tokens:.2f}")
    if doc_token_counts:
        print(f"Min Tokens: {min(doc_token_counts)}")
        print(f"Max Tokens: {max(doc_token_counts)}")
    print("-" * 30)

if __name__ == "__main__":
    process_and_count_tokens()
