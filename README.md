# ♟️ Chess AI Agent using Machine Learning

> Diploma Thesis – University of Patras  
> Department of Electrical & Computer Engineering  
> Author: **Georgios Zisis**  
> Supervisor: **Evangelos Dermatas**

## 📖 Overview

This project explores the development of an intelligent chess-playing agent using Machine Learning and Deep Learning techniques. The aim was to investigate different neural network architectures, board representations, and training approaches while operating on consumer-grade hardware.

The project evolved through three major versions, progressively improving performance against traditional chess engines such as Stockfish.

---

## 🎯 Objectives

- Develop a chess-playing AI agent from scratch.
- Evaluate various state representations.
- Compare CNN and Residual Neural Network architectures.
- Analyze performance under hardware limitations.
- Explore AlphaZero-inspired Monte Carlo Tree Search techniques.

---

## 🧠 State of the Art

This work was inspired by modern chess engines:

- **Stockfish** – combines alpha-beta search with neural network evaluation.
- **AlphaZero** – learns exclusively through self-play.
- **Leela Chess Zero (Lc0)** – open-source AlphaZero-inspired engine.
- **AlphaZero.jl** – lightweight Julia implementation of AlphaZero principles.

---

## ⚙️ Hardware & Software

### Hardware

- AMD Ryzen 7 5800H
- NVIDIA GeForce RTX 3060
- 16GB RAM

### Software

- Python
- PyTorch
- CUDA
- Stockfish Chess Engine
- Lichess Elite Database

---

## 📊 Dataset

### Data Source

The training data was collected from the **Lichess Elite Database**.

### Version 1 & 2

- Games from players rated 2400+
- Opponents rated 2200+
- PGN files until 2015
- ~2.5 million moves

### Version 3

- PGN files until 2017
- ~25 million moves

### Data Pipeline

Custom utilities were developed for:

- Loading PGN files
- Combining multiple PGN datasets
- Creating CSV datasets
- Dataset preprocessing and management

---

## 🔄 Chess Position Representation

### Board State (Input)

Positions are initially represented using **FEN notation**.

#### Version 1

Input tensor:

```text
12 × 8 × 8
```

Represents the locations of all chess pieces.

#### Version 2

Input tensor:

```text
18 × 8 × 8
```

Includes:

- Piece locations
- Side to move
- Castling rights
- En passant information

### Move Representation (Output)

A fixed action space was created:

```text
4544 possible moves
```

Every move is mapped to a unique index, including illegal moves for fixed-size network outputs.

---

## 🏗️ Model Architectures

### Version 1 – CNN

Architecture:

```text
Input (12x8x8)
↓
Conv Layer (32 filters)
↓
Conv Layer (64 filters)
↓
Conv Layer (64 filters)
↓
Flatten
↓
Fully Connected (512)
↓
Output (4544)
```

Features:

- Kernel size = 3
- Padding = 1
- ReLU activation functions

---

### Version 2 – Residual Neural Network

#### Convolutional Block

```text
18 Input Planes
↓
256 Feature Maps
↓
Batch Normalization
↓
ReLU
```

#### Residual Tower

```text
6 Residual Blocks
256 Filters Each
```

#### Policy Head

Predicts move probabilities:

```text
256 → 2 feature maps
↓
BatchNorm
↓
ReLU
↓
Flatten
↓
FC Layer
↓
4544 Outputs
```

#### Value Head

Predicts board evaluation:

```text
256 → 1 feature map
↓
BatchNorm
↓
ReLU
↓
Flatten
↓
FC Layer
↓
Tanh
↓
1 Output Value
```

---

## 🚀 Training

### Hyperparameters

```python
LOSS_FUNCTION = CrossEntropyLoss()
OPTIMIZER = Adam(lr=0.001)
BATCH_SIZE = 256
```

### Epochs

| Dataset Size | Epochs |
| ------------ | ------ |
| 2.5 Million  | 100    |
| 25 Million   | 50     |

### Training Times

| Mode           | Time      |
| -------------- | --------- |
| Offline (2.5M) | ~2 hours  |
| Online (2.5M)  | ~6 hours  |
| Online (25M)   | ~30 hours |

### Loss Improvement

| Model  | Initial Loss | Final Loss |
| ------ | ------------ | ---------- |
| CNN    | 4.15         | 2.30       |
| ResNet | 1.39         | 0.97       |

---

## 🌳 Monte Carlo Tree Search

An AlphaZero-inspired MCTS implementation was developed.

### Node Structure

Each node stores:

- Current state
- Current player
- Parent node
- Child nodes
- Prior probability
- Visit count
- Node value

### Search Process

1. Selection
2. Expansion
3. Backpropagation

Unlike classical MCTS, rollout simulations were omitted, following the AlphaZero methodology.

---

## 📈 Experimental Results

### Version 1 (CNN)

✅ Produces mostly reasonable moves

✅ Competitive against low-level Stockfish

❌ Struggles to convert winning positions

❌ Weak endgame performance

---

### Version 2 (Residual Network)

✅ Stronger than Version 1

✅ Achieves wins against Stockfish

✅ Better strategic understanding

❌ Still struggles in endgames

❌ Aggressive style occasionally causes inaccuracies

---

### Version 3 (25M Dataset)

✅ Win rate greater than 50% against multiple Stockfish configurations

✅ Average Centipawn Loss between 30–50

✅ Significantly improved endgame performance

✅ Consistently stronger than previous versions

⚠️ Still tends to over-prioritize pawn promotion opportunities

---

## ⚠️ AlphaZero Limitations

Although AlphaZero-style self-play was successfully tested in simpler environments such as:

- Tic-Tac-Toe
- Connect Four

full chess self-play training proved impractical on consumer hardware because:

- MCTS is extremely time-consuming.
- One game may require up to an hour.
- Massive amounts of self-play data are needed.
- Multiple training iterations are required to reach strong performance.

---

## 🔮 Future Work

Possible future improvements include:

- Larger datasets
- Endgame-focused training data
- More efficient MCTS implementation
- C++ or Julia implementation for speed
- Better search algorithms
- Full reinforcement learning pipeline

---

## 🏁 Conclusion

This project demonstrates that a supervised-learning chess agent can achieve a respectable playing level using commercially available hardware.

The final model:

- Can defeat intermediate human players.
- Can defeat low-level Stockfish configurations.
- Produces moves similar to those recommended by Stockfish 18.
- Shows clear improvement with larger datasets and stronger architectures.

Despite the limitations of reinforcement learning on consumer hardware, the project successfully achieved its primary objective of building a capable chess AI agent.

---

## 👨‍🎓 Academic Information

**University of Patras**  
Department of Electrical & Computer Engineering

**Diploma Thesis**

**Construction of an Automated Chess Engine Using Machine Learning Methods**

**Author:** Georgios Zisis  
**Supervisor:** Evangelos Dermatas
