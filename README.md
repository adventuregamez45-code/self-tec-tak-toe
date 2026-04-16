# Self-Playing Tic-Tac-Toe & Connect 4 AI

A neural network AI that learns to play Tic-Tac-Toe and Connect 4 
through self-play, trained with policy gradient reinforcement learning.

## How to Run

**Tic-Tac-Toe:**
1. `python train.py` — trains the neural network (~20-50k episodes)
2. `python play.py` — play against the AI

**Connect 4:**
1. `python connect4_train.py`
2. `python connect4_play.py`

## Requirements
pip install torch numpy

## Features
- 3 difficulty levels (Easy / Medium / Hard)
- Hard mode is unbeatable — uses win/block/fork detection + neural net
- CNN architecture for Connect 4
- Self-play + random opponent training
