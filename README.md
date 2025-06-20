# Chess Engine Project

This is a personal chess engine project developed in Python, starting from basic classical search and evolving toward supervised learning and AlphaZero-style reinforcement learning.

## Goals

- ✅ Build a basic chess engine using classical algorithms (minimax, alpha-beta)
- ✅ Implement board evaluation functions
- ⏳ Add supervised learning to predict strong moves from grandmaster games
- ⏳ Experiment with self-play and AlphaZero-style learning

## Tech Stack

- Python 3
- [python-chess](https://pypi.org/project/python-chess/) – for board representation and legal moves
- PyTorch – for training neural networks
- Logging / Markdown – for tracking progress and thesis write-up

## Directory Structure
- src/ → Engine code (main loop, eval functions, etc.)
- logs/ → Auto-generated log files
- data/ → Training data (e.g., PGN files)
- notebooks/ → ML experiments
- chess_engine_log.md → Dev journal for thesis