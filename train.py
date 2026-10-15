"""Train Aira AI on any .txt file.

Kids story mode (default dataset = TinyStories):
    python train.py

Your own text:
    python train.py --data_path mybook.txt --keep_unicode      (e.g. Malayalam)
"""
import argparse
import math
import os
import sys
import urllib.request
from dataclasses import asdict

import torch

from model import AiraLM, Config

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
)


def get_args():
    p = argparse.ArgumentParser(description="Train Aira AI")
    p.add_argument("--data_path", default="tinystories.txt")
    p.add_argument("--max_chars", type=int, default=8_000_000, help="use only first N chars")
    p.add_argument("--keep_unicode", action="store_true", help="don't strip non-ASCII (needed for Malayalam)")
    p.add_argument("--out", default="aira_ai_best.pt")
    p.add_argument("--resume", action="store_true", help="continue from --out checkpoint")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_iters", type=int, default=6000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def load_text(args):
    if not os.path.exists(args.data_path):
        if args.data_path != "tinystories.txt":
            sys.exit(f"File not found: {args.data_path}")
        print("📥 Downloading TinyStories (kids stories, ~20 MB)...")
        try:
            urllib.request.urlretrieve(TINYSTORIES_URL, args.data_path)
        except Exception as e:
            sys.exit(f"❌ Download failed ({e}). Check internet or put a .txt file here manually.")
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read(args.max_chars * 2)
    text = text.replace("<|endoftext|>", "\n\n")
    if not args.keep_unicode:
        text = text.encode("ascii", "ignore").decode()
    return text[: args.max_chars]


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Aira AI trainer on: {device.upper()}")

    text = load_text(args)
    print(f"📚 Loaded {len(text):,} characters")

    ckpt = None
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=device)
        chars = ckpt["chars"]
        cfg = Config(**ckpt["config"])
        print("♻️  Resuming from checkpoint")
    else:
        chars = sorted(set(text))
        cfg = Config(len(chars), args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout)

    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"🔤 Vocab size: {len(chars)}")

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i : i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = AiraLM(cfg).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):  # linear warmup, then cosine decay
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.max_iters - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for s in ("train", "val"):
            out[s] = sum(model(*get_batch(s))[1].item() for _ in range(args.eval_iters)) / args.eval_iters
        model.train()
        return out

    best = ckpt["val_loss"] if ckpt else float("inf")
    for step in range(args.max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            l = estimate()
            print(f"Step {step:>5} | train {l['train']:.4f} | val {l['val']:.4f} | lr {lr_at(step):.2e}")
            if l["val"] < best:
                best = l["val"]
                torch.save(
                    {"model": model.state_dict(), "config": asdict(cfg), "chars": chars, "val_loss": best},
                    args.out,
                )
                print(f"  ✅ saved {args.out} (val {best:.4f})")

        _, loss = model(*get_batch("train"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print("🏁 Training complete! Now run:  python aira_story.py")


if __name__ == "__main__":
    main()
