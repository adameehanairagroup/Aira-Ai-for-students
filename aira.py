#!/usr/bin/env python3
"""Aira - Ollama-style runner for Aira AI. Pure NumPy, works in Termux.

    python aira.py list                 # show models (*.npz next to this file)
    python aira.py run                  # interactive story time in the terminal
    python aira.py serve                # web UI + API at http://localhost:11435
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANIMALS = ["puppy", "kitten", "elephant", "bunny", "monkey", "lion", "frog", "panda"]
ENCOURAGE = [
    "Yum yum! 😋 One more bite and the story continues!",
    "Wow, big bite! 🌟 Aira likes it!",
    "Open wide... here comes the next part! 🚂",
    "Chew chew chew! 🐿️ Story is waiting!",
]
BLOCKLIST = {"kill", "die", "dead", "death", "blood", "gun", "hate", "stupid", "hell", "damn"}


# ───────────────────────── model (NumPy inference with KV cache) ─────────────────────────
class Aira:
    def __init__(self, path):
        d = np.load(path)
        meta = json.loads(str(d["_meta"]))
        c = meta["config"]
        self.chars = meta["chars"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.W = {k: d[k] for k in d.files if k != "_meta"}
        self.C, self.nh, self.T, self.L = c["n_embd"], c["n_head"], c["block_size"], c["n_layer"]
        self.hs = self.C // self.nh
        self.K = np.zeros((self.L, self.T, self.nh, self.hs), np.float32)
        self.V = np.zeros_like(self.K)
        self.n = 0

    @staticmethod
    def _ln(x, w, b):
        return (x - x.mean()) / np.sqrt(x.var() + 1e-5) * w + b

    def step(self, tok):
        """Feed one token, return logits for the next one."""
        W, C, nh, hs, pos = self.W, self.C, self.nh, self.hs, self.n
        x = W["tok.weight"][tok] + W["pos.weight"][pos]
        for i in range(self.L):
            p = f"blocks.{i}."
            h = self._ln(x, W[p + "ln1.weight"], W[p + "ln1.bias"])
            qkv = W[p + "attn.qkv.weight"] @ h
            q = qkv[:C].reshape(nh, hs)
            self.K[i, pos] = qkv[C : 2 * C].reshape(nh, hs)
            self.V[i, pos] = qkv[2 * C :].reshape(nh, hs)
            s = (self.K[i, : pos + 1] * q).sum(-1) / np.sqrt(hs)       # (t, nh)
            a = np.exp(s - s.max(0))
            a /= a.sum(0)
            o = (a[:, :, None] * self.V[i, : pos + 1]).sum(0).reshape(C)
            x = x + W[p + "attn.proj.weight"] @ o + W[p + "attn.proj.bias"]
            h = self._ln(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
            h = W[p + "mlp.0.weight"] @ h + W[p + "mlp.0.bias"]
            h = 0.5 * h * (1 + np.tanh(0.79788456 * (h + 0.044715 * h**3)))  # GELU
            x = x + W[p + "mlp.2.weight"] @ h + W[p + "mlp.2.bias"]
        x = self._ln(x, W["ln_f.weight"], W["ln_f.bias"])
        self.n += 1
        return W["tok.weight"] @ x

    def _prime(self, ids):
        self.n = 0
        logits = None
        for t in ids:
            logits = self.step(t)
        return logits

    def generate(self, prompt, max_chars, temperature, top_k):
        """Yield characters one by one."""
        hist = [self.stoi[c] for c in prompt if c in self.stoi][-(self.T // 2):] or [0]
        logits = self._prime(hist)
        for _ in range(max_chars):
            if self.n >= self.T:                       # context full: re-prime with recent tokens
                logits = self._prime(hist[-(self.T // 2):])
            l = logits / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, len(l))
                l = np.where(l < np.partition(l, -k)[-k], -np.inf, l)
            p = np.exp(l - l.max())
            t = int(np.random.choice(len(p), p=p / p.sum()))
            hist.append(t)
            yield self.chars[t]
            logits = self.step(t)


# ───────────────────────── story helpers ─────────────────────────
def list_models():
    return sorted(glob.glob(os.path.join(HERE, "*.npz")))


def build_prompt(text):
    text = (text or "").strip() or random.choice(ANIMALS)
    if text.lower().startswith("once upon"):
        return text
    return f"Once upon a time, there was a little {text.rstrip('.')}."


def make_story(model, prompt, g):
    """Generate a full story; retry if it contains a blocked word."""
    for _ in range(5):
        out = prompt
        for ch in model.generate(prompt, g.max_chars, g.temperature, g.top_k):
            out += ch
            if len(out) > g.min_chars and out.endswith("\n\n"):
                break
        out = out.strip()
        if not (set(re.findall(r"[a-z]+", out.lower())) & BLOCKLIST):
            return out
    return None


def paced_words(story):
    for w in story.split(" "):
        yield w + " "


# ───────────────────────── CLI: run ─────────────────────────
def cmd_run(model, g, speed):
    print("\n🧸✨ Aira Story Time ✨🧸")
    print("Type an animal (e.g. bunny) or 'bunny named Mia'. Enter = surprise. /bye to quit.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/bye", "/exit", "q"):
            break
        print("🤔 Aira is thinking...")
        story = make_story(model, build_prompt(text), g)
        if story is None:
            print("Oops, Aira got confused. Try again! 🙈\n")
            continue
        sentences = 0
        for w in paced_words(story):
            print(w, end="", flush=True)
            time.sleep(speed)
            if w.strip().endswith((".", "!", "?")):
                sentences += 1
                if sentences % 4 == 0:
                    print("\n\n   " + random.choice(ENCOURAGE) + "\n")
                    time.sleep(1.2)
        print("\n\n🌟 The End! 🌟\n")
    print("Bye bye! 👋😊")


# ───────────────────────── server: web UI + API ─────────────────────────
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aira Story Time</title><style>
body{font-family:system-ui,sans-serif;background:#fff7e6;margin:0;padding:16px;color:#3b2f2f}
h1{text-align:center;font-size:28px;margin:8px 0 14px}
#animals{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
#animals button{font-size:34px;padding:10px 0;border-radius:16px;border:3px solid transparent;background:#fff;cursor:pointer}
#animals button.sel{border-color:#ff9800;background:#ffe0b2}
input{width:100%;box-sizing:border-box;font-size:18px;padding:12px;margin:12px 0;border-radius:12px;border:2px solid #ddd}
#go{width:100%;font-size:22px;padding:14px;border:0;border-radius:16px;background:#ff9800;color:#fff;font-weight:700}
#go:disabled{opacity:.5}
#out{white-space:pre-wrap;font-size:22px;line-height:1.6;background:#fff;border-radius:16px;padding:16px;margin-top:14px;min-height:120px}
</style></head><body>
<h1>🧸 Aira Story Time</h1>
<div id="animals"></div>
<input id="name" placeholder="Name of the little one (optional)">
<button id="go">✨ Tell me a story!</button>
<div id="out">Pick a friend and press the button 👆</div>
<script>
const A=[["🐶","puppy"],["🐱","kitten"],["🐘","elephant"],["🐰","bunny"],["🐵","monkey"],["🦁","lion"],["🐸","frog"],["🐼","panda"]];
const CHEER=["Yum yum! 😋 One more bite!","Wow, big bite! 🌟","Open wide... next part! 🚂","Chew chew chew! 🐿️"];
let pick="puppy";const box=document.getElementById("animals");
A.forEach(([e,n])=>{const b=document.createElement("button");b.textContent=e;b.onclick=()=>{pick=n;[...box.children].forEach(x=>x.classList.remove("sel"));b.classList.add("sel")};box.appendChild(b)});
box.children[0].classList.add("sel");
const go=document.getElementById("go"),out=document.getElementById("out");
go.onclick=async()=>{
  out.textContent="🤔 Aira is thinking...";go.disabled=true;
  const nm=document.getElementById("name").value.trim();
  try{
    const r=await fetch("/api/generate",{method:"POST",body:JSON.stringify({prompt:pick+(nm?" named "+nm:"")})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf="",s=0,first=true;
    while(true){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf("\n"))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(!line.trim())continue;
        const j=JSON.parse(line);if(first){out.textContent="";first=false}
        out.textContent+=j.response;
        if(/[.!?]\s*$/.test(j.response)&&++s%4==0)out.textContent+="\n\n   "+CHEER[Math.floor(Math.random()*4)]+"\n\n";}}
    out.textContent+="\n\n🌟 The End! 🌟";
  }catch(e){out.textContent="Oops, something went wrong 🙈"}
  go.disabled=false;
};
</script></body></html>"""


def cmd_serve(model, g, speed, host, port):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/tags":
                names = [{"name": os.path.basename(p)} for p in list_models()]
                self._send(200, json.dumps({"models": names}), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/generate":
                return self._send(404, "not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._send(400, "bad json", "text/plain")
            with lock:  # one story at a time (model keeps a KV cache)
                story = make_story(model, build_prompt(req.get("prompt")), g)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                if story is None:
                    story = "Oops, Aira got confused. Try again! 🙈"
                for w in paced_words(story):
                    self.wfile.write((json.dumps({"response": w, "done": False}) + "\n").encode())
                    self.wfile.flush()
                    time.sleep(speed)
                self.wfile.write((json.dumps({"response": "", "done": True}) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    print(f"🧸 Aira is serving at  http://localhost:{port}   (Ctrl+C to stop)")
    if host == "0.0.0.0":
        print("   Other devices on your Wi-Fi can open http://<this-phone-ip>:%d" % port)
    try:
        ThreadingHTTPServer((host, port), H).serve_forever()
    except KeyboardInterrupt:
        print("\nBye bye! 👋")


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(prog="aira", description="Ollama-style runner for Aira AI")
    ap.add_argument("cmd", choices=["list", "run", "serve"])
    ap.add_argument("--model", help="path to .npz (default: first *.npz next to aira.py)")
    ap.add_argument("--speed", type=float, default=0.12, help="seconds per word")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--min_chars", type=int, default=250)
    ap.add_argument("--max_chars", type=int, default=900)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on Wi-Fi")
    ap.add_argument("--port", type=int, default=11435)
    a = ap.parse_args()

    if a.cmd == "list":
        ms = list_models()
        print("\n".join(f"{os.path.basename(m)}\t{os.path.getsize(m)/1e6:.1f} MB" for m in ms) or "No .npz models found.")
        return
    path = a.model or (list_models() or [None])[0]
    if not path or not os.path.exists(path):
        sys.exit("No model found. Put aira.npz next to aira.py (made with export_numpy.py).")
    print(f"📦 Loading {os.path.basename(path)} ...")
    model = Aira(path)
    g = SimpleNamespace(**{k: getattr(a, k) for k in ("temperature", "top_k", "min_chars", "max_chars")})
    if a.cmd == "run":
        cmd_run(model, g, a.speed)
    else:
        cmd_serve(model, g, a.speed, a.host, a.port)


if __name__ == "__main__":
    main()
