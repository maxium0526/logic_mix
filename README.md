# LogicMix and 2WayAug-PL Official Implementation
This readme is to be done.

## Requirements

It may be better to use `uv` to run this code. It will automatically install dependencies before running the code.

## Usage

1. Edit `config.py` to choose the dataset (remember to change the dataset path). Change the parameters if necessary.
2. Run `uv run train.py` to train a classification model.
3. Run `uv run evaluate.py` to evaluate the trained model.

## LogicMix Class

The LogicMix algorithm is implemented as a subclass of `torch.utils.data.Dataset`. It should be simple to use by warping a training set.

