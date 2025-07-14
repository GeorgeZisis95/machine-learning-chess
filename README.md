# ♟️ Chess AI with Supervised Learning

This project implements a simple yet effective chess-playing AI using supervised learning. It is built with PyTorch and trained on high-ELO human games from the Lichess Elite Database. The system is capable of predicting high-quality moves from board positions and generating full games that can be visualized and evaluated on Lichess.

---

## 📌 Project Goals

- Build a neural network that plays chess using only supervised learning (no search or reinforcement learning).
- Train it to predict the most likely human move given a board position.
- Evaluate its quality through top-k accuracy and game-level performance.

---

## 🧠 Model Architecture

The model, `ChessNet`, is a lightweight convolutional neural network (CNN) that maps board positions to move probabilities:
- Input: 12×8×8 tensor (binary planes representing piece type and color).
- Convolutional layers: [Conv2D → ReLU] × 3
- Fully connected layers: Flatten → Linear(512) → Output(4544)
- Output: A probability distribution over 4,544 possible UCI moves.

## 🧪 Training Details

- Loss function: Cross-entropy Loss
- Optimizer: Adam (lr=0.001)
- Batch size: Tunable (commonly 64 or 128)
- Device: CUDA-enabled GPU
- Data: Lichess PGN files (Elite games), parsed and filtered
- Encoding: FEN → binary planes

Training logs include both training and validation loss, which are saved and visualized after each epoch.

## 📈 Evaluation
### 🔢 Quantitative

- Top-1 Accuracy: ~35%
- Top-3 Accuracy: ~58%
- Average Centipawn Loss: ~180 (on Lichess analysis)

Demonstrates understanding of opening and positional play, with most errors occurring in complex tactical positions.

### 🎮 Qualitative (Lichess PGN Visualization)

The model can generate complete games by playing against itself or a random/opponent agent. These games are saved in .pgn format and imported to Lichess for engine-assisted evaluation.

Example Lichess analysis:
- 4 inaccuracies
- 4 mistakes
- 14 blunders
- Accuracy scores: 53% vs 59%

## 🔁 Move Selection Logic
During inference, the model samples from the top-k predicted moves (usually top-3) to increase diversity and reduce deterministic repetition. This allows the agent to play varied, realistic games rather than repeating a single deterministic path.

## 🧰 Notebooks & Tools
- train.ipynb: Prepares the dataset and trains the model.
- eval.ipynb: Evaluates top-k accuracy and plays full games.
- Games can be exported to PGN and visualized on lichess.org/paste.

## 🤝 Acknowledgments
- Lichess for their open access to high-quality chess game data.
- The python-chess library for FEN/PGN parsing and board manipulation.
- PyTorch for the deep learning framework.

## 📚 Future Work
- Add value head to estimate board evaluation.
- Introduce reinforcement learning or AlphaZero-style self-play.
- Train on filtered, high-quality or blunder-free datasets.
- Experiment with deeper architectures or attention mechanisms.
- Add GUI or web interface for human vs AI play.