from transformers import GPT2Tokenizer

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    phrase = "Artificial intelligence is metamorphosing the world!"
    print("PHRASE 1:", phrase)

    # Tokens (strings)
    tokens = tokenizer.tokenize(phrase)
    print("\nTokens:")
    print(tokens)

    # IDs
    token_ids = tokenizer.encode(phrase)
    print("\nToken IDs:")
    print(token_ids)

    # Decode each ID individually (sanity check)
    print("\nDétails par token (ID -> decode):")
    for tid in token_ids:
        txt = tokenizer.decode([tid])
        print(tid, repr(txt))

    # Phrase 2 with long word
    phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."
    print("\n\nPHRASE 2:", phrase2)

    tokens2 = tokenizer.tokenize(phrase2)
    print("\nTokens phrase2:")
    print(tokens2)

    # Focus on the long word (simple extraction by locating "antidisestablishmentarianism" piece-by-piece)
    # Heuristic: print tokens that contain fragments of it
    print("\nSous-tokens liés au mot long (heuristique):")
    long_related = [t for t in tokens2 if "anti" in t or "dis" in t or "establish" in t or "arian" in t]
    print(long_related)

if __name__ == "__main__":
    main()
