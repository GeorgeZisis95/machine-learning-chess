# Development Documentation

This document collects theory, explanations, and notes related to the chess engine project.

---

## Table of Contents

1. [Python-Chess Library](#python-chess-library)  
2. [Chess Engine Fundamentals](#chess-engine-fundamentals)  
3. [Supervised Learning](#supervised-learning)  
4. [Reinforcement Learning](#reinforcement-learning)  
5. [Neural Networks](#neural-networks)  
6. [AlphaZero Concepts](#alphazero-concepts)  
7. [Additional Notes & References](#additional-notes--references)  

---

## Python-Chess Library

### Overview

*Write your understanding of the library and how you use it.*

### Key Classes and Methods

- `chess.Board()`
- `board.legal_moves`
- `board.push(move)`
- `board.pop()`
- `chess.Move`
- `board.fen()`
- `chess.pgn.read_game()`

### Notes

*Add anything specific you discover or want to remember.*

---

## Chess Engine Fundamentals

### Basic Concepts

- Board representation
- Move generation
- Evaluation functions (material count, positional evaluation)
- Search algorithms (minimax, alpha-beta pruning)

---

## Supervised Learning

### Core Ideas

- Input features and labels
- Loss functions (Cross-Entropy, MSE)
- Training and validation sets

### Application to Chess

- Predicting moves from grandmaster games
- Input encoding (board states as tensors)
- Output encoding (move indices)

---

## Reinforcement Learning

### Key Concepts

- Agents, states, actions, rewards
- Policy networks and value networks
- Self-play and Monte Carlo Tree Search (MCTS)

---

## Neural Networks

### Architecture Types

- Fully connected (Dense) layers
- Convolutional Neural Networks (CNNs)
- Residual Networks (ResNets)

### Training Details

- Optimizers (Adam, SGD)
- Learning rate schedules
- Regularization (dropout, weight decay)

---

## AlphaZero Concepts

- Combining supervised learning and reinforcement learning
- Policy and value networks
- MCTS guided by neural networks
- Self-play data generation

---

## Additional Notes & References

- Links to papers, tutorials, videos
- Personal reflections
- Questions to research further
