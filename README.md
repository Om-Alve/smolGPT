# SMOL-GPT 🦾

A minimal PyTorch implementation for training your own small LLM from scratch. Designed for educational purposes and simplicity, featuring efficient training, flash attention, and modern sampling techniques.

## Features ✨

- **Minimal Codebase**: Pure PyTorch implementation with no abstraction overhead
- **Modern Architecture**: GPT model with:
  - Flash Attention (when available)
  - RMSNorm and SwiGLU
  - Efficient top-k/p/min-p sampling
  - Rotary embeddings - RoPE (Optional)
- **Training Features**:
  - Mixed precision (bfloat16/float16)
  - Gradient accumulation
  - Learning rate decay with warmup
  - Weight decay & gradient clipping
- **Dataset Support**: Built-in TinyStories dataset processing
- **Custom Tokenizer**: SentencePiece tokenizer training integration

## Installation 🛠️

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+ with CUDA
- Modern GPU (recommended)

## Quick Start 🚀

### Option 1: Full Training Cycle

1. **Prepare Dataset**
```bash
python preprocess.py prepare-dataset --vocab-size 4096
```

2. **Start Training**
```bash
python train.py
```

*Training and validation loss are logged in `out/logs/`. To visualize using TensorBoard, run:*
```bash
tensorboard --logdir=/out/logs
```

3. **Generate Text**
```bash
python sample.py \
    --prompt "Once upon a time" \
    --num_samples 3 \
    --temperature 0.7 \
    --max_new_tokens 500
```

### Option 2: Use Pre-trained Model

1. **Download Assets**
```bash
# Download tokenizer
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/tok4096.model -P data/

# Download pre-trained checkpoint
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/ckpt.pt -P out/
```

2. **Run Inference**
```bash
python sample.py \
    --prompt "Once upon a time" \
    --tokenizer_path data/tok4096.model \
    --ckpt_path out/ckpt.pt \
    --num_samples 3 \
    --max_new_tokens 200 \
    --temperature 0.7
```

## Pre-trained Model Details 🔍

The provided checkpoint was trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Architecture:
- 4096-token vocabulary
- 8 heads
- 8-layer transformer
- 512 embedding dimension
- Trained on `~4 Billion Tokens` for around `18.5` hours

Validation Loss - `1.0491`

![Loss Curve](assets/loss.png)

## Sample Outputs 📝

### Example 1
```text
Prompt: One day, Lily met a unicorn

Output:
One day, Lily met a unicorn in the park. The unicorn had shiny fur and a pretty dress. Lily asked the unicorn, "Where did you come from?"
The unicorn replied, "I came from the forest and wanted to meet you, Lily. I am here to make sure you are safe."
Lily and the unicorn played together and had lots of fun. But then, the unicorn started to act funny. Lily didn't know what was happening. Suddenly, the unicorn turned into a little girl and said, "I was under a spell, and your kindness broke it. Thank you for breaking it."
Lily was surprised and happy. She gave the unicorn a big hug and they played together all day. From that day on, the unicorn would always come to play with Lily - her new friend.
```

```
Prompt: The dragon flew over the mountains

Output:
The dragon flew over the mountains, over the rivers and over the rivers. He was very brave and strong.
One day, the dragon saw something very strange. It was a big, shiny rock. He wanted to know what it was, so he flew down and touched it with his nose. Suddenly, the rock began to move!
The dragon was so surprised! He had never seen anything like it before. He looked around and saw that it was a little mouse! The mouse was very scared and started to run away.
The dragon was very sad. He wanted to help the mouse, so he decided to try and make friends. He flew around and around until he found the mouse. He said hello to the mouse and asked if he wanted to be friends.
The mouse was so happy! He said yes, and they played together all day long. From then on, the dragon and the mouse were the best of friends. They had lots of fun together and the dragon was never lonely again.
```

## Configuration ⚙️

Key parameters (modify in `config.py`):

**Model Architecture**:
```python
GPTConfig(
    block_size=512,    # Context length
    n_layer=8,         # Number of transformer layers
    n_head=8,          # Number of attention heads
    n_embed=512,       # Embedding dimension
    dropout=0.2,       # Dropout rate
    bias=False,        # Use bias in layers
    use_rotary=False,  # Toggle rotary embeddings
)
```

**Training**:
```python
TrainingConfig(
    batch_size=64,
    max_iters=30000,
    learning_rate=6e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_iters=1000
)
```

## File Structure 📁

```
om-alve-smolgpt/
├── config.py       - Model & training configuration
├── dataset.py      - Data loading & preprocessing
├── model.py        - GPT model implementation
├── preprocess.py   - Dataset preparation scripts
├── sample.py       - Text generation script
├── tokenizer.py    - Tokenizer wrapper
└── train.py        - Main training loop
```

## Contributing 🤝

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- Performance improvements
- New features

---

### Training RIG SPECS (Rented via LightningAI)  
- **GPU**: NVIDIA L4 Tensor Core (Optimized for AI workloads)  
- **vCPUs**: 16  
- **RAM**: 64 GB  
- **VRAM**: 24 GB  
---

**Note**: This implementation is inspired by modern LLM training practices and adapted for educational purposes. For production use, consider scaling up model size and dataset.
