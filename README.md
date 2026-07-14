♟️ Chess AI Agent using Machine Learning

Diploma Thesis – University of Patras
Department of Electrical & Computer Engineering
Author: Georgios Zisis
Supervisor: Evangelos Dermatas
Based on the thesis"Construction of an Automated Chess Engine Using Machine Learning Methods"

📖 Overview
This project explores the development of an intelligent chess-playing agent using modern Machine Learning and Deep Learning techniques. The objective was to investigate different neural network architectures, data representations, and optimization methods while keeping computational requirements within the capabilities of consumer-grade hardware.
The project was implemented in three major versions, progressively improving the playing strength of the agent against traditional chess engines such as Stockfish.

🎯 Goals
The main goals of this project were:

Develop a chess-playing AI agent from scratch.
Compare different neural network architectures.
Investigate board-state representations.
Evaluate performance under limited computational resources.
Explore AlphaZero-inspired approaches and Monte Carlo Tree Search (MCTS).,

🧠 Related Work
The project draws inspiration from modern chess engines including:

Stockfish – combines alpha-beta search with neural evaluation.
AlphaZero – learns entirely through self-play.
Leela Chess Zero (Lc0) – open-source AlphaZero-inspired engine.
AlphaZero.jl – lightweight Julia implementation of AlphaZero principles.

⚙️ Hardware & Software
Hardware

AMD Ryzen 7 5800H
NVIDIA GeForce RTX 3060
16GB RAM

Software

Python
PyTorch
CUDA
Lichess Elite Database
Stockfish Chess Engine

📊 Dataset
Training data was collected from the Lichess Elite Database.
Version 1 & 2

Games between players rated 2400+ vs 2200+
PGN files up to 2015
Approximately 2.5 million moves

Version 3

PGN files up to 2017
Approximately 25 million moves

Data Pipeline
Custom utilities were created for:

Loading PGN files
Creating datasets
Generating CSV training data
Managing multiple PGN sources simultaneously

🔄 State Representation
The board state is initially represented using FEN notation.
Version 1
Input tensor:
12 × 8 × 8

Representing piece positions only.
Version 2
Input tensor:
18 × 8 × 8

Including:

Piece positions
Side to move
Castling rights
En Passant information

Action Representation
All legal and illegal moves were mapped to a fixed output space of:
4544 possible actions

Each move corresponds to a unique index.

🏗️ Model Architectures
Version 1 – Convolutional Neural Network (CNN)

3 Convolutional Layers
ReLU Activations
Flatten Layer
Fully Connected Layers
Output Size: 4544 moves

Version 2 – Residual Neural Network (ResNet)
Convolutional Block

18 Input Planes
256 Feature Maps
Batch Normalization
ReLU Activation

Residual Tower

6 Residual Blocks
256 Filters per Layer

Policy Head

Predicts move probabilities
Output: 4544 moves

Value Head

Predicts board evaluation
Output: Single scalar value using Tanh activation

🚀 Training
Hyperparameters
Python1Loss Function = Cross Entropy2Optimizer = Adam3Learning Rate = 0.0014Batch Size = 256Show more lines
Training schedule:

100 epochs for 2.5M positions
50 epochs for 25M positions

Training Times

ModeDatasetTimeOffline2.5M positions~2 hoursOnline2.5M positions~6 hoursOnline25M positions~30 hours

Results During Training
Loss reduction achieved:

ModelInitial LossFinal LossCNN4.152.30ResNet1.390.97

🌳 Monte Carlo Tree Search (MCTS)
An AlphaZero-inspired MCTS implementation was developed.
Core Components
Node
Stores:

Current State
Current Player
Parent Node
Children Nodes
Prior Probability
Visit Count
Node Value

Tree Search
Implements:

Selection
Expansion
Backpropagation

Following the AlphaZero methodology without rollout simulations.

📈 Experimental Results
Version 1 (CNN)
✅ Produces mostly sensible moves
✅ Plays competitively against low-level Stockfish
❌ Struggles to convert winning positions
❌ Weak endgame performance

Version 2 (ResNet)
✅ Stronger than Version 1
✅ Achieves occasional victories against Stockfish
✅ Better positional understanding
❌ Still struggles in endgames
❌ More aggressive and sometimes unstable play style

Version 3 (Large Dataset)
✅ Consistently defeats lower Stockfish levels
✅ Win rates exceeding 50% in several tests
✅ Average Centipawn Loss around 30–50
✅ Significant improvement in endgame conversion
⚠️ Still exhibits an excessive preference for pawn promotion strategies

⚠️ AlphaZero Feasibility
Although AlphaZero-style self-play was successfully tested on simpler environments such as:

Tic-Tac-Toe
Connect Four

it was impractical for full chess training on consumer hardware due to:

Extremely long search times
Massive self-play requirements
Repeated training cycles needed for convergence

🔮 Future Work
Potential improvements include:

Larger training datasets
Endgame-focused datasets
Faster MCTS implementations
C++ or Julia reimplementation for performance
Improved search strategies
Full-scale reinforcement learning training,

🏁 Conclusion
This thesis successfully demonstrates that a machine learning-based chess agent can achieve a respectable playing level using supervised learning and commercially available hardware. The final model can defeat intermediate human players and low-level Stockfish configurations while following move choices that align closely with modern chess engine evaluations.
