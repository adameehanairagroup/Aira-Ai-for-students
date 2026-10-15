"""Aira Story Time 🧸 - a playful story teller for kids (great for meal time!)

    python aira_story.py
    python aira_story.py --speed 0.05     # slower typing, easier to read aloud
"""
import argparse
import random
import re
import sys
import time

import torch

from model import AiraLM, Config

ANIMALS = {
    "1": ("🐶", "puppy"), "2": ("🐱", "kitten"), "3": ("🐘", "elephant"),
    "4": ("🐰", "bunny"), "5": ("🐵", "monkey"), "6": ("🦁", "lion"),
    "7": ("🐸", "frog"), "8": ("🐼", "panda"),
}
NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anu", "Kiko", "Pinky", "Bubu", "Aira"]
ENCOURAGE = [
    "Yum yum! 😋 One more bite and the story continues!",
    "Wow, big bite! 🌟 Aira likes it!",
    "Open wide... here comes the next part! 🚂",
    "Chew chew chew! 🐿️ Story is waiting!",
]
# Tiny safety net: stop and retry if a story contains any of these words.
BLOCKLIST = {"kill", "die", "dead", "death", "blood", "gun", "hate", "stupid", "hell", "damn"}


def is_clean(text: str) -> bool:
    return not (set(re.findall(r"[a-z]+", text.lower())) & BLOCKLIST)


def load(path, device):
    ck = torch.load(path, map_location=device)
    model = AiraLM(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    chars = ck["chars"]
    return model, {c: i for i, c in enumerate(chars)}, chars


def make_story(model, stoi, chars, device, prompt, args):
    seed = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long, device=device)
    out = prompt
    for tok in model.stream(seed, args.max_chars, args.temperature, args.top_k):
        out += chars[tok]
        # end the story at a paragraph break once it's long enough
        if len(out) > args.min_chars and out.endswith("\n\n"):
            break
    return out.strip()


def typewriter(story, speed):
    """Print slowly, word by word, with a cheer every few sentences."""
    sentences = 0
    for word in story.split(" "):
        print(word, end=" ", flush=True)
        time.sleep(speed)
        if word.endswith((".", "!", "?")):
            sentences += 1
            if sentences % 4 == 0:
                print("\n\n   " + random.choice(ENCOURAGE) + "\n")
                time.sleep(1.2)
    print("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="aira_ai_best.pt")
    ap.add_argument("--speed", type=float, default=0.12, help="seconds per word")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--min_chars", type=int, default=250)
    ap.add_argument("--max_chars", type=int, default=900)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, stoi, chars = load(args.ckpt, device)
    except FileNotFoundError:
        sys.exit("No trained model found. Run `python train.py` first.")

    print("\n🧸✨ Welcome to AIRA STORY TIME! ✨🧸\n")
    while True:
        print("Pick a friend for the story:")
        for k, (emoji, name) in ANIMALS.items():
            print(f"  {k}) {emoji} {name}")
        print("  r) 🎲 Surprise me      q) 👋 Bye bye")
        choice = input("\nYour choice: ").strip().lower()
        if choice == "q":
            print("Bye bye! See you at next meal! 👋😊")
            break
        if choice not in ANIMALS:
            choice = random.choice(list(ANIMALS))
        emoji, animal = ANIMALS[choice]

        name = input("Name of the little one (Enter to skip): ").strip() or random.choice(NAMES)
        prompt = f"Once upon a time, there was a little {animal} named {name}."
        if not all(c in stoi for c in prompt):
            prompt = f"Once upon a time, there was a little {animal}."

        print(f"\n{emoji} Aira is thinking of a story...\n")
        story = None
        for _ in range(5):  # retry if the story isn't clean
            s = make_story(model, stoi, chars, device, prompt, args)
            if is_clean(s):
                story = s
                break
        if story is None:
            print("Oops, Aira got confused. Let's try again! 🙈\n")
            continue

        print("📖 " + "─" * 40 + "\n")
        typewriter(story, args.speed)
        print("🌟 The End! 🌟\n")
        if input("Another story? (Enter = yes, n = no): ").strip().lower() == "n":
            print("Bye bye! 👋😊")
            break


if __name__ == "__main__":
    main()
