<div align="center">

<img src="assets/logo.png" alt="Aira Group" width="280">

# 🧸 Aira AI

**A tiny story-telling AI for kids. Trained from scratch, runs on a phone.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/Runs%20on-NumPy%20only-013243?logo=numpy&logoColor=white)](#-run-on-your-phone-termux)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/adameehanaira/Aira-Ai-for-students?label=model)](https://github.com/adameehanaira/Aira-Ai-for-students/releases/latest)

[**Download model**](https://github.com/adameehanaira/Aira-Ai-for-students/releases/latest) ·
[**Train in Colab**](https://colab.research.google.com/github/adameehanaira/Aira-Ai-for-students/blob/main/notebooks/Aira_AI_Colab_v2.ipynb) ·
[**Report a bug**](https://github.com/adameehanaira/Aira-Ai-for-students/issues)

</div>

---

## ✨ What is Aira AI?

Aira AI is a small **GPT-style language model** (2.7 million parameters) that tells short, playful stories. It was built for one simple moment: **a child at meal time**. Pick a friend (🐶 🐰 🐘 🐼), type a name, and Aira tells a story word by word while cheering the little one on with *"Yum yum! 😋 One more bite!"*

It is also a complete, readable example of how a language model is built, trained and shipped, from raw text to a web app running on a phone. Perfect for students.

> 🧪 **Honest note:** Aira is a very small character-level model. Stories are cute but often silly: names can change halfway through and logic is loose. That is part of the fun, and part of what makes it a good learning project.

## 📖 Sample story

> *Once upon a time, there was a little bunny named Mia. She loved to play games. One day, she saw a circle on a rod. It was her favorite dress. She wanted to make a magical painting with the rainbow shape.*

*(Real output from the released model, `temperature=0.75`.)*

## 🚀 Features

- 🧠 **Trained from scratch**: a transformer written in plain PyTorch (about 100 lines), no pretrained weights
- 🧸 **Kids story mode**: animal friends, custom names, typewriter effect, meal-time cheers
- 📱 **Runs on Android (Termux)**: pure NumPy inference with KV cache, **no PyTorch needed**
- 🌐 **Ollama-style interface**: `aira list`, `aira run`, `aira serve` with a kid-friendly web UI and a streaming API
- 🔤 **Train on any text**: English, Malayalam (`--keep_unicode`), code, names, anything in a `.txt`
- ☁️ **One-click Colab notebook**: train your own in about 25 minutes on a free GPU
- 🛡️ **Basic safety filter**: blocked-word check with automatic retry

## 📊 Model card

| | |
|---|---|
| **Architecture** | Decoder-only transformer, 6 layers, 6 heads, 192 dim, context 128 |
| **Parameters** | 2,721,024 |
| **Tokenizer** | Character-level (79 symbols) |
| **Training data** | First 8M characters of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| **Training** | 15,000 steps, AdamW, warmup + cosine decay, batch 64, 1× T4 GPU |
| **Validation loss** | **0.775** |
| **Files** | `aira_ai_best.pt` (10.3 MB, PyTorch) · `aira.npz` (9.6 MB, NumPy) |

## ⚡ Quick start

### 1. Train your own (Colab, free GPU)

Open [`notebooks/Aira_AI_Colab_v2.ipynb`](notebooks/Aira_AI_Colab_v2.ipynb) in Google Colab, switch to a **T4 GPU**, and choose **Runtime → Run all**. The trained model is saved to your Google Drive.

### 2. Or use the released model on a computer

```bash
git clone https://github.com/adameehanaira/Aira-Ai-for-students.git
cd Aira-Ai-for-students
pip install -r requirements.txt
```

Download `aira_ai_best.pt` from [Releases](https://github.com/adameehanaira/Aira-Ai-for-students/releases/latest), place it in this folder, then:

```bash
python aira_story.py
```

### 3. Or train locally

```bash
python train.py                                    # downloads TinyStories, trains ~6000 steps
python train.py --max_iters 15000 --batch_size 64  # the released settings
python train.py --data_path my_book.txt            # your own text
python train.py --data_path malayalam.txt --keep_unicode
```

Every hyperparameter is a flag. Run `python train.py --help` to see them all.

## 📱 Run on your phone (Termux)

PyTorch is hard to install on a phone, so Aira ships a **NumPy-only** runner.

```bash
pkg update
pkg install python python-numpy
```

Put `aira.py` and `aira.npz` (from [Releases](https://github.com/adameehanaira/Aira-Ai-for-students/releases/latest)) in one folder:

```bash
python aira.py list      # show available models
python aira.py run       # story time in the terminal
python aira.py serve     # web UI at http://localhost:11435
```

Open **http://localhost:11435** in your phone browser for the kid-friendly story page. Use `--host 0.0.0.0` to let a TV or tablet on the same Wi-Fi open it too.

**Tip:** `termux-wake-lock` keeps the server alive when the screen is off.

### API

`aira serve` exposes an Ollama-style streaming endpoint:

```bash
curl -N -X POST http://localhost:11435/api/generate -d '{"prompt": "bunny named Mia"}'
```

Each line is a JSON object: `{"response": "word ", "done": false}`.

## 🗂️ Project structure

```
Aira-Ai-for-students/
├── model.py            # transformer (PyTorch)
├── train.py            # training script, all settings are CLI flags
├── aira_story.py       # kids story mode (PyTorch)
├── export_numpy.py     # .pt → .npz (for phones)
├── aira.py             # NumPy runner: list / run / serve
├── notebooks/
│   └── Aira_AI_Colab_v2.ipynb
├── assets/             # logo and images
├── requirements.txt
└── LICENSE
```

## 🔧 How it works

1. **Text → numbers**: each character becomes an ID (79 in total).
2. **Transformer**: token + position embeddings go through 6 blocks of causal self-attention and an MLP (pre-LayerNorm, residual connections, flash attention, weight tying).
3. **Training**: the model learns to predict the next character, over and over, on millions of characters of children's stories.
4. **Generation**: sample one character at a time (temperature + top-k) until the story ends.
5. **Phone inference**: `export_numpy.py` converts the weights to `.npz`; `aira.py` re-implements the forward pass in NumPy with a KV cache so each new character is fast.

## ⚠️ Limitations and safety

- Small character-level model: **expect silly sentences** and names that change mid-story.
- The blocked-word filter is a **basic safety net, not a guarantee**. Please **supervise children** when they use it.
- Trained on English stories only; other languages need your own dataset.

## 🗺️ Roadmap

- [ ] BPE (word-piece) tokenizer for more coherent stories
- [ ] Larger model trained on the full TinyStories set
- [ ] Malayalam story model
- [ ] Keep character names consistent through a story
- [ ] Read-aloud (text-to-speech) mode

Ideas and pull requests are welcome. 🙌

## 🙏 Credits

- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** by Ronen Eldan and Yuanzhi Li (Microsoft Research). Please check the dataset page for its license terms.
- Inspired by **Andrej Karpathy's** *nanoGPT* and *"Let's build GPT"*.

## 👤 Author

<table>
  <tr>
    <td align="center" width="170">
      <img src="assets/author.png" alt="Author" width="130"><br>
    </td>
    <td>
      <b><a href="https://github.com/adameehanaira">@adameehanaira</a></b><br>
      Creator of Aira AI · <b>Aira Group</b><br>
      <i>Code the Higher Tomorrow.</i><br><br>
      Built Aira AI as a learning project for students, from first character to a model running on a phone.
    </td>
  </tr>
</table>

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">

If Aira made you smile, **give the repo a ⭐**. It helps other students find it!

<sub>Made with ❤️ by Aira Group</sub>

</div>
