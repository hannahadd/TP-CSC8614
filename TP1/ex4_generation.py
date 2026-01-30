import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")


def decode(out_ids):
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def section(title):
    print("\n" + "=" * 10, title, "=" * 10)


def main():
    print("SEED =", SEED)
    print("PROMPT =", repr(prompt))

    # 1) Greedy decoding
    section("GREEDY (max_length=50)")
    out_greedy = model.generate(
        **inputs,
        max_length=50,
        do_sample=False
    )
    print(decode(out_greedy))

    # 2) Sampling: temperature=0.7, top_k=50, top_p=0.95 (5 seeds)
    section("SAMPLING (temp=0.7, top_k=50, top_p=0.95) - 5 runs")
    def generate_once(seed, repetition_penalty=None, temperature=0.7):
        torch.manual_seed(seed)
        kwargs = dict(
            **inputs,
            max_length=50,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
        )
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        out = model.generate(**kwargs)
        return decode(out)

    for s in [1, 2, 3, 4, 5]:
        print("SEED", s)
        print(generate_once(s))
        print("-" * 40)

    # 3) Repetition penalty comparison (same seed)
    section("REPETITION PENALTY (seed=1, temp=0.7, top_k=50, top_p=0.95)")
    print("Sans penalty:")
    print(generate_once(seed=1, repetition_penalty=None))
    print("\nAvec penalty (repetition_penalty=2.0):")
    print(generate_once(seed=1, repetition_penalty=2.0))

    # 4) Temperature extremes (keep top_k/top_p fixed, same seed)
    section("TEMPERATURE EXTREMES (seed=1, top_k=50, top_p=0.95)")
    print("Temp=0.1:")
    print(generate_once(seed=1, temperature=0.1))
    print("\nTemp=2.0:")
    print(generate_once(seed=1, temperature=2.0))

    # 5) Beam search (num_beams=5)
    section("BEAM SEARCH (num_beams=5, max_length=50)")
    t0 = time.perf_counter()
    out_beam5 = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    t1 = time.perf_counter()
    print(decode(out_beam5))
    print(f"Time num_beams=5: {t1 - t0:.3f}s")

    # 6) More beams + timing
    section("BEAM SEARCH TIMING (num_beams=10 then 20)")
    for nb in [10, 20]:
        t0 = time.perf_counter()
        _ = model.generate(
            **inputs,
            max_length=50,
            num_beams=nb,
            early_stopping=True
        )
        t1 = time.perf_counter()
        print(f"Time num_beams={nb}: {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
