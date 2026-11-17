# CS188 Project 5 — Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Course](https://img.shields.io/badge/Course-UC%20Berkeley%20CS188-orange)](https://inst.eecs.berkeley.edu/~cs188/)

A student implementation of **Project 5 – Machine Learning** from UC Berkeley's **CS188: Introduction to Artificial Intelligence**. The project explores six core machine-learning topics, from linear classification to convolutional networks and self-attention, using Python and PyTorch.

## Project Overview

| Question | Topic | What it does |
|---|---|---|
| **Q1** | Perceptron | Learns a linear decision boundary for binary classification and updates its weights when it makes a mistake. |
| **Q2** | Regression | Trains a neural network to approximate a nonlinear function, such as \(\sin(x)\). |
| **Q3** | Digit Classification (MLP) | Classifies 28×28 MNIST handwritten-digit images into 10 classes (0–9) using fully connected layers. |
| **Q4** | Language Identification (RNN) | Reads a word character by character and predicts which of five languages it belongs to. |
| **Q5** | Convolutional Neural Network (CNN) | Implements a 2D convolution operation and uses the extracted image features for digit classification. |
| **Q6** | Self-Attention | Implements scaled dot-product attention with Query, Key, Value and a causal mask that blocks access to future tokens. |

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.x** — models, automatic differentiation and training
- **NumPy** — numerical and dataset operations
- **Matplotlib** — plots and training visualizations
- **Tkinter** — desktop demo interface (included with standard Windows Python installations)
- **Pillow** — optional image-processing support in the demo

The project can be tested on **CPU**; a CUDA-capable GPU is not required. The datasets are already included in the repository, so no separate dataset download is needed.

## Getting Started

### 1. Install prerequisites

Install:
- [Python 3.10 or newer](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads/)

On Windows, enable **Add Python to PATH** during installation. Verify the tools:

```bash
python --version
git --version
```

### 2. Clone the repository

```bash
git clone https://github.com/24020364-NguyenHuuVinh/project-5-CS-188-13.git
cd project-5-CS-188-13
```

### 3. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python packages

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib pillow
```

Verify the installation:

```bash
python -c "import torch, numpy, matplotlib; print('PyTorch:', torch.__version__)"
```

> **Notes**
> - The repository does not currently provide a `requirements.txt`; the command above installs the dependencies used by the project.
> - Pillow is optional for some demo image features. The core Autograder requires PyTorch, NumPy and Matplotlib.
> - Tkinter is normally included with Python on Windows. On Debian/Ubuntu, install it with `sudo apt install python3-tk` if the demo reports that Tkinter is missing.
> - For a CUDA-enabled installation, use the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and choose the configuration matching your system. CPU is sufficient for the commands below.

## Run the Project

Run commands from the repository root, with the virtual environment activated.

### Run all questions (Q1–Q6)

```bash
python autograder.py --no-graphics
```

### Run one question at a time

```bash
python autograder.py -q q1 --no-graphics
python autograder.py -q q2 --no-graphics
python autograder.py -q q3 --no-graphics
python autograder.py -q q4 --no-graphics
python autograder.py -q q5 --no-graphics
python autograder.py -q q6 --no-graphics
```

Remove `--no-graphics` to enable the visualizations supported by the Autograder. Use `--no-graphics` for terminal-only runs or automated environments.

### Launch the desktop demo

```bash
python demo.py
```

The Tkinter application provides interactive demonstrations for **Q1–Q5**. It is an additional learning and visualization tool; Q6 is tested separately through the Autograder.

### Optional: train and save MLP/CNN checkpoints

```bash
python train_checkpoints.py
```

This script trains the Q3 MLP and Q5 CNN models, evaluates them and saves their weights under `checkpoints/`.

### Optional: run the Character-level GPT experiment

```bash
python chargpt.py
```

This is an extension experiment built around the Q6 attention implementation. It trains a small character-level text-generation model using `input.txt`; it is **not required** to run or grade Q1–Q6.

## Repository Structure

```text
project-5-CS-188-13/
├── autograder.py          # Public tests for Q1–Q6
├── backend.py             # Dataset classes, data loading and visualizations
├── models.py              # Perceptron, MLP, RNN, CNN and Attention models
├── train.py               # Training loops for Q1–Q5
├── losses.py              # Loss functions
├── demo.py                # Desktop demo for Q1–Q5
├── train_checkpoints.py   # Train and save MLP/CNN weights
├── gpt_model.py           # Transformer block and Character GPT model
├── chargpt.py             # Optional character-level text-generation experiment
├── input.txt              # Text corpus for the GPT experiment
├── data/
│   ├── mnist.npz          # Handwritten-digit dataset
│   └── lang_id.npz        # Language-identification dataset
├── checkpoints/           # Saved MLP/CNN checkpoints
├── ui_weights/            # Model weights used by the desktop demo
└── README.md
```

## Notes

- Run commands from the repository root so the data files can be located correctly.
- The included files in `data/` are used by the project; do not delete them.
- Local Autograder results are useful for development, but they do not replace the course's official submission and grading process.

## References

- [UC Berkeley CS188 — Project 5](https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/)
- [PyTorch — Installation Guide](https://pytorch.org/get-started/locally/)
- [Python — Downloads](https://www.python.org/downloads/)
