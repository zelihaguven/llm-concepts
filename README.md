# llm-concepts

Hands-on notebooks for **From Scratch to Scale**, my Medium series on how large language models work and why they are expensive to run. Each post builds one idea from scratch in PyTorch, checks it against a reference implementation, and measures it.

![From Scratch to Scale, Part 1: Attention](assets/part1_attention.png)

## Posts and notebooks

| Part | Post | Notebook |
| --- | --- | --- |
| 1 | [Attention, and Why Transformers Are Expensive](MEDIUM_LINK_PART_1) | [`01_transformer_attention.ipynb`](notebooks/architecture/01_transformer_attention.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelihaguven/llm-concepts/blob/main/notebooks/architecture/01_transformer_attention.ipynb) |
| 2 | Latent attention and the KV-cache *(coming next)* | |
| 3 | Mixture of Experts *(planned)* | |

Later parts will move from model architecture to embeddings (GTEs, Matryoshka Representation Learning) and retrieval (query expansion, RAG Fusion).

## What Part 1 covers

- Scaled dot-product attention in about 20 lines, verified against `torch.nn.functional.scaled_dot_product_attention`
- Why scores are divided by √d, measured on random vectors
- The causal mask and multi-head attention, verified against `nn.MultiheadAttention`
- Inside GPT-2 small: the one head that links "it" to "animal", and the attention sink on the first token
- The quadratic cost of attention: time and memory against sequence length

## Running the notebooks

The easiest way is the **Open in Colab** button: the first cell clones this repo, so nothing needs to be installed.

To run locally (everything runs on a CPU):

```bash
git clone https://github.com/zelihaguven/llm-concepts.git
cd llm-concepts
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

GPT-2 small (about 500 MB) is downloaded from Hugging Face the first time the notebook runs. Figures are saved to `figures/`.

## Repository layout

```
llm-concepts/
├── src/                  # shared, reusable code (e.g. attention.py)
├── notebooks/
│   └── architecture/     # one notebook per post
├── assets/               # images used in this README
├── figures/              # created when the notebooks run
└── requirements.txt
```

## Feedback

I'm a computer engineering student learning these topics in public. If you spot a mistake or have a suggestion, please open an issue.

## License

[MIT](LICENSE)
