# 🧠 Chess AI using Supervised Learning

This project builds a neural network that learns to play chess using supervised learning (SL). The model is trained to predict the best move from a given board position, based on real game data.

## ✅ Features

- PGN parser to extract (FEN, move) training pairs
- FEN encoder → 8×8×12 tensor input
- UCI move → classification output (4672 possible moves)
- PyTorch `Dataset` and `DataLoader`
- CNN-based policy network
- Full training pipeline with CrossEntropyLoss


## 📊 Model Architecture

- Input: 12×8×8 tensor (board state)
- 3 convolutional layers + 2 FC layers
- Output: 4672-class logits (move prediction)

## 🧪 Training

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Trained on real FEN → move pairs from PGN files

## 🎯 Goal

To build a solid supervised learning chess agent that can:
- Play without blundering
- Handle full games well
- Learn strong, human-like moves

## 📌 Future Plans

- Train on larger dataset (100k+ positions)
- Improve model architecture (BatchNorm, Dropout, deeper convs)
- Play against Stockfish and evaluate
- Optional GUI or CLI agent to play live games
