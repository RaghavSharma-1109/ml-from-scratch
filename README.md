# ML From Scratch

Machine learning algorithms implemented from scratch using Python and NumPy — no scikit-learn, no black boxes. The goal is to understand what's actually happening inside these algorithms, not just call `.fit()`.

## Implemented

### Linear Regression
- Built from scratch using NumPy — gradient descent, no library shortcuts
- Trained and evaluated on a real dataset (not synthetic/toy data)
- Data shuffled before train/test split using index arrays, with explicit verification that feature-target pairs stayed correctly aligned after the shuffle
- Model loss compared against a mean-baseline predictor to confirm the model was actually learning something, not just guessing the average

### Result
The model beat the mean baseline, though the margin was modest — most likely due to unnormalized features and untuned hyperparameters (learning rate, iterations) rather than a flaw in the core algorithm. Next iteration will include feature normalization and a hyperparameter sweep to see how much headroom is actually there.

## Why this exists
Part of a self-directed backend/applied-ML internship prep track. This repo specifically focuses on building intuition for the math and mechanics under the hood before relying on libraries that hide them.

## Structure
