## 🗓️ Day 1 – [20-06-2025]

**Goal**: Initialize the board and play a random legal move.

**Steps Taken**:
- Installed `python-chess`
- Loaded the default board
- Printed the board
- Selected and played a random legal move

**Code Snippet**:
```python
import chess, random

board = chess.Board()
print("Start:\n", board)

move = random.choice(list(board.legal_moves))
board.push(move)

print("After move:\n", board)
```

## 📎 Appendix A: Python Environment Setup
This appendix details the environment configuration used to ensure reproducibility and consistency across development and training phases.

### 📦 Python Environment
- Python version: 3.13.3
- Environment manager: venv
- Libraries: python-chess, torch, optionally jupyter, matplotlib

### 🛠️ Creating and Managing the Environment

1. Create a virtual environment using: 
`python -m venv .venv`
2. Activate the environment on windows with: 
`.venv\Scripts\activate` 
3. Upgrade pip using:
`pip install --upgrade pip`
4. Install required packages:
    - `pip install python-chess`
    - `pip install torch`