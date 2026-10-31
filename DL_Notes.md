# Deep Learning Training — Day 1

## Learning goals

Day 1 introduces neural-network foundations and builds toward the Transformer: how text becomes tokens and vectors, why context matters, and how self-attention combines information. The session also surveys common model families and discusses practical interview preparation.

## Topics covered

- Perceptrons, artificial neural networks (ANNs), multilayer perceptrons (MLPs), parameters, and hyperparameters
- Convolutional neural networks (CNNs), including VGG16 and ResNet
- Recurrent neural networks (RNNs)
- A visual introduction to NanoGPT and the Transformer architecture
- Industry interview scenarios, including differences between traditional software and AI interviews
- ImmverseAI internship interview discussion

Suggested study balance: spend roughly 20% of study time writing code and 80% building conceptual understanding, then use coding to check that understanding.

## How a language model processes text

A neural network operates on numbers. Before text can be processed, a tokenizer divides it into tokens and maps those tokens to integer IDs. Depending on the tokenizer, a token can be a word, part of a word, punctuation, or another text unit.

For a teaching example, split `I love deep learning` into four tokens:

| Token | Example ID |
|---|---:|
| I | 1 |
| love | 2 |
| deep | 3 |
| learning | 4 |

These IDs are labels, not measurements: ID 4 is not inherently larger or more meaningful than ID 1. Real tokenizers use much larger vocabularies and often subword tokens. They also define special tokens for tasks such as marking sequence boundaries or padding.

### A small vocabulary example

```python
tokens = ["I", "love", "deep", "learning"]
vocabulary = {token: index for index, token in enumerate(tokens, start=1)}
token_ids = [vocabulary[token] for token in tokens]
```

This toy vocabulary only handles the words it contains. A lookup for an unknown word would fail unless the code adds an unknown-token policy. Production tokenizers handle unseen words through subword tokenization and special-token rules.

## Embeddings

An embedding maps each token ID to a learned vector. The following values are invented only to illustrate the shape of the representation; individual dimensions do not have simple, fixed human-readable meanings.

| Token | Dimension 1 | Dimension 2 | Dimension 3 |
|---|---:|---:|---:|
| king | 0.8 | 0.6 | 0.2 |
| queen | 0.7 | 0.7 | 0.2 |
| apple | 0.1 | 0.2 | 0.9 |
| banana | 0.2 | 0.3 | 0.8 |

Token IDs identify entries; embeddings provide trainable numerical representations. In a batch containing one sentence with four tokens and three-dimensional embeddings, the embedding output has shape `(1, 4, 3)`: batch size, sequence length, embedding width.

## Why attention?

The meaning of a token depends on its context. Attention lets tokens build relationships with other tokens in the sequence. RNNs process sequences step by step and carry a hidden state forward; self-attention instead computes relationships across sequence positions directly, which supports parallel processing during training.

### Query, key, and value

For an input matrix `X`, a self-attention layer forms learned projections:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

`W_Q`, `W_K`, and `W_V` are learned weight matrices. Queries and keys determine which positions should receive attention; values contain the information that gets combined. These are learned vector representations, not manually assigned semantic labels.

Scaled dot-product attention is commonly written as:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

The dot products produce compatibility scores. Scaling by `√d_k` helps control score magnitude; softmax turns each row of scores into non-negative weights summing to one; multiplying by `V` computes a weighted combination of value vectors. In causal language models, a mask prevents a position from attending to future tokens.

### Search analogy

Imagine a student searching a digital library. The student's question resembles a query, document descriptors resemble keys, and document contents resemble values. The system scores matches and combines useful content. This is an intuition for the roles of Q, K, and V, not a literal description of every search engine or attention implementation.

## Practical exercises

1. Build a small token-to-ID vocabulary and inspect the resulting IDs.
2. Use a TensorFlow embedding layer and check that a batch of one four-token sentence produces shape `(1, 4, 3)` when the embedding width is three. Actual initialized values vary.
3. Create `attention_practical.py` with small Q, K, and V matrices. Compute `QKᵀ`, apply softmax along the key-position axis, and multiply the resulting weights by V. Inspect the score, weight, and output shapes.
4. Compare the result with a hand-calculated example to confirm that the weights sum to one and that the output is a weighted combination of V.

## Broader model landscape

- **CNNs:** use local filters to learn spatial patterns, commonly for image tasks.
- **RNNs:** process a sequence recurrently, carrying state from one step to the next.
- **Transformers:** use attention and feed-forward layers to process token representations.
- **GANs:** train a generator to create samples and a discriminator to distinguish generated samples from real ones.
- **Autoencoders:** encode an input into a compact latent representation and decode it back into a reconstruction.
- **RAG:** retrieves relevant external passages and supplies them to a language model to ground its response.

### GAN architecture example (28 × 28 grayscale images)

**Generator:** random noise (100 values) → dense layer → reshape to `7 × 7 × 128` → transposed-convolution layers → generated image `28 × 28 × 1`.

**Discriminator:** image `28 × 28 × 1` → convolution and LeakyReLU layers → flatten → dense output → sigmoid probability for real versus generated.

Layer sizes and activations vary by implementation; this is a conceptual outline, not a complete specification.

### Autoencoder summary

For a `28 × 28` image, flattening produces 784 input values. An encoder might reduce dimensions such as `784 → 256 → 64`, then store a compact latent code `z` (for example, 2–32 values). A decoder expands `z` back toward 784 reconstructed pixels. The target is the input itself, so labels are not required; a reconstruction loss such as mean squared error trains encoder and decoder together.

The bottleneck encourages a compact representation. After training, the encoder can produce features for visualization or downstream tasks. Autoencoders can support compression, denoising, anomaly detection, inpainting, and feature learning. A small bottleneck may lose detail, and plain autoencoders do not generally provide a reliable way to sample realistic new images; variational autoencoders address generative modeling with a different latent-space objective.

## Local model exploration with Ollama

Ollama is software for running and managing language models locally. A suggested exploration setup is VS Code, Ollama, and the `gemma2:2b` model. Availability and performance depend on the machine and the installed Ollama version.

## Environment setup notes

Use a project virtual environment and install only the dependencies needed for the exercise. For example, the repository's `requirements-ml-notebooks.txt` covers the TensorFlow notebook examples, while `requirements-llm.txt` is for the optional LangChain/Ollama experiments. PyTorch CPU installation instructions are platform-specific; follow the official PyTorch selector for the current command rather than combining unrelated package lists into one environment.

## Python refresher

- Python is a high-level, interpreted language designed for readable code.
- Common built-in types include integers, floats, strings, lists, tuples, dictionaries, and booleans.
- `if` / `elif` / `else` provide conditional flow; `for` and `while` provide loops.
- Define reusable functions with `def`.

## Further study in this repository

- [NanoGPT-style notebook](../../notebooks/mini_gpt.ipynb)
- [GAN notebook](../../notebooks/GAN.ipynb)
- [Transformer paper](../../papers/NIPS-2017-Attention-is-all-you-need-Paper.pdf)
- [RAG survey](../../papers/Retrieval-Augmented Generation for Large Language Models- A Survey.pdf)
