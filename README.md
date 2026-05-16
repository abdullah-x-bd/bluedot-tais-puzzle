# BlueDot Technical AI Safety Puzzle 1

This repository contains the notebook and architecture image for BlueDot Technical AI Safety Puzzle 1.

The task is to investigate a text classifier trained on eight binary features and identify which feature is not represented linearly at the specified hidden layer activation.

## Files

- `puzzle.ipynb` contains the puzzle notebook.
- `model_architecture.png` contains the architecture diagram.

## Quick start

Open `puzzle.ipynb` and run the setup cell. The notebook will install `sentence-transformers` and `torch`, then clone the upstream puzzle repo used by the notebook.

## Original upstream source used inside the notebook

`https://github.com/SamDower/bluedot-tais-puzzle`

## Notes

The notebook expects model and data files from the upstream puzzle repository:

- `model.pt`
- `data/train.jsonl`
- `data/test.jsonl`
- `feature_names.json`
