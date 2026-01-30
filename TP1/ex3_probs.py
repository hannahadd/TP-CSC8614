import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def token_probs_for_phrase(model, tokenizer, phrase: str):
    inputs = tokenizer(phrase, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab)

    probs = torch.softmax(logits, dim=-1)
    input_ids = inputs["input_ids"][0]

    print("PHRASE:", phrase)
    print("Tokens:", tokenizer.tokenize(phrase))
    print("\nP(token_t | tokens_<t>) for t>=1:")
    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        p = probs[0, t - 1, tok_id].item()
        tok_txt = tokenizer.decode([tok_id])
        print(t, repr(tok_txt), f"{p:.3e}")

    return logits, inputs


def logp_and_ppl_from_logits(logits, input_ids):
    log_probs = torch.log_softmax(logits, dim=-1)

    total_logp = 0.0
    n = 0
    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        lp = log_probs[0, t - 1, tok_id].item()
        total_logp += lp
        n += 1

    avg_neg_logp = -(total_logp / n)
    ppl = math.exp(avg_neg_logp)
    return total_logp, avg_neg_logp, ppl


def topk_next_tokens(model, tokenizer, prefix: str, k: int = 10):
    inp = tokenizer(prefix, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
        logits2 = out.logits  # (1, seq_len, vocab)

    # distribution for the NEXT token = last time step
    last_logits = logits2[0, -1, :]
    last_probs = torch.softmax(last_logits, dim=-1)

    vals, idx = torch.topk(last_probs, k=k)
    print("\nTOP", k, "next tokens for prefix:", repr(prefix))
    for p, tid in zip(vals.tolist(), idx.tolist()):
        print(repr(tokenizer.decode([tid])), f"{p:.3e}")


def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 1) probs token-by-token for the base sentence
    phrase1 = "Artificial intelligence is fascinating."
    logits1, inputs1 = token_probs_for_phrase(model, tokenizer, phrase1)
    ids1 = inputs1["input_ids"][0]
    total_logp1, avg_neg_logp1, ppl1 = logp_and_ppl_from_logits(logits1, ids1)
    print("\nTOTAL logp:", total_logp1)
    print("AVG neg logp:", avg_neg_logp1)
    print("PERPLEXITY:", ppl1)

    # 2) compare with scrambled order
    phrase2 = "Artificial fascinating intelligence is."
    logits2, inputs2 = token_probs_for_phrase(model, tokenizer, phrase2)
    ids2 = inputs2["input_ids"][0]
    total_logp2, avg_neg_logp2, ppl2 = logp_and_ppl_from_logits(logits2, ids2)
    print("\nTOTAL logp:", total_logp2)
    print("AVG neg logp:", avg_neg_logp2)
    print("PERPLEXITY:", ppl2)

    # 3) french sentence
    phrase3 = "L'intelligence artificielle est fascinante."
    logits3, inputs3 = token_probs_for_phrase(model, tokenizer, phrase3)
    ids3 = inputs3["input_ids"][0]
    total_logp3, avg_neg_logp3, ppl3 = logp_and_ppl_from_logits(logits3, ids3)
    print("\nTOTAL logp:", total_logp3)
    print("AVG neg logp:", avg_neg_logp3)
    print("PERPLEXITY:", ppl3)

    # 4) top-10 next tokens
    topk_next_tokens(model, tokenizer, "Artificial intelligence is", k=10)

    print("\n--- Summary (perplexities) ---")
    print("ppl(phrase1):", ppl1)
    print("ppl(phrase2):", ppl2)
    print("ppl(phrase3):", ppl3)


if __name__ == "__main__":
    main()
