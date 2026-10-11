# Neural networks from scratch

I wrote these during my A-levels (2024-25) to learn how neural networks train. Everything is NumPy, with
no ML framework. Keras is only used to download the datasets.

I started with Harrison Kinsley's (sentdex) [Neural Networks from Scratch](https://nnfs.io) videos, and
`neuron.py`, `objects.py` and `dataset.py` come from those. The series wasn't finished, so I moved to
Samson Zhang's video,
[Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU). `mnist.py`
follows his code closely, including a bias-gradient bug I've since fixed (see Fixed since). The later
scripts build on it: mini-batches and the menu in `mnistBatches.py`, a second hidden layer in
`threeLayer.py`, and the CNN in `convolution.py`.

## Results

These come from retraining in 2026, after the fixes listed under Fixed since. Each network was trained
for 100 epochs with a learning rate of 0.1 and a batch size of 64, once with each of the seeds 0, 1 and
2. Accuracy is measured on all 10,000 Fashion-MNIST test images.

| Script | Network | Mean test accuracy | Range over seeds | Saved weights (seed 0) |
|---|---|---|---|---|
| `mnistBatches.py` | 784-10-10: one hidden layer of 10 ReLU units | 84.4% | 83.7 to 84.8% | `variablesRetrained.json`, 83.7% |
| `mnistBatches.py`, `hiddenUnits=20` | 784-20-10: one hidden layer of 20 ReLU units | 85.4% | 85.2 to 85.8% | not saved |
| `threeLayer.py` | 784-20-20-10: two hidden layers of 20 ReLU units | 86.1% | 85.7 to 86.4% | `threeVariableRetrained.json`, 86.3% |

I didn't record the settings I used in 2024, so these are a reconstruction: 100 epochs, 0.1 and 64 are
the values in a commented-out training call in the code. I chose seed 0 for the saved weights before
training. On my machine, retraining with seed 0 reproduces the saved weights exactly. Another machine's
maths library may round differently in the last decimal places.

The original 2024 weights are still here: `variables.json` scores 84.0% and `threeVariable.json` scores
85.4%. Their settings and random starting weights weren't recorded, so they can't be reproduced.

### Did fixing the bias gradient change accuracy?

Not measurably. I retrained both networks with the same seeds and settings, but with the old bias
gradient put back. They averaged 84.8% (784-10-10) and 86.0% (784-20-20-10). For each seed, the fixed
and buggy versions were within 0.7 points of each other, while different seeds of the same version
varied by up to 1.1 points. For the smaller network the buggy version scored slightly higher on all
three seeds, but three seeds can't separate a difference that small from chance.

The biases are a small part of each network (20 of 7,960 parameters in the smaller one, 50 of 16,330 in
the larger), and the weights were always trained correctly, so they made up for it. The accuracy never
showed the bug; a gradient check did.

### Does the second hidden layer help?

784-20-10 and 784-20-20-10 differ only in the second hidden layer, so comparing them isolates depth.
The second layer added 0.7 points on average (85.4% to 86.1%) for only 420 extra parameters. The two
ranges overlap slightly (85.2 to 85.8% against 85.7 to 86.4%), so with three seeds each this is
suggestive rather than settled. For comparison, widening the single hidden layer from 10 to 20 units
added 1.0 point (84.4% to 85.4%) and doubled the number of parameters.

The 784-20-10 runs call `gradientDescent` in `mnistBatches.py` with `hiddenUnits=20`; the menu only
trains the 10-unit version.

## What's in here

- `neuron.py`, `objects.py`, `dataset.py`: my first experiments, following the sentdex videos. A layer
  worked out by hand, then dense layers, ReLU, softmax and cross-entropy loss written as classes and
  tried on a synthetic spiral dataset.
- `mnist.py`: one hidden layer on MNIST digits with full-batch gradient descent, following Samson
  Zhang's video. It prints training accuracy, then tests on the 10,000 MNIST test images (83.5% with
  seed 0).
- `mnistBatches.py`: one hidden layer on Fashion-MNIST with shuffled mini-batches. An interactive menu
  lets you retrain, test a single image, or run the whole test set. The original weights are in
  `variables.json` and the retrained ones in `variablesRetrained.json`. `gradientDescent` takes
  `hiddenUnits` (10 by default) to change the width of the hidden layer.
- `threeLayer.py`: the same with two hidden layers. The original weights are in `threeVariable.json`
  and the retrained ones in `threeVariableRetrained.json`.
- `gradientCheck.py`: checks the backward pass of `mnistBatches.py` and `threeLayer.py`. It works out
  each gradient a second way, by nudging every parameter up and down and measuring the change in the
  loss, and prints the relative error against backpropagation for each parameter array. Below about
  1e-7 means backpropagation is right; every array currently comes out around 1e-10.
- `convolution.py`: a small convolutional network, trained one image at a time with a decaying learning
  rate:
  - two 9x9 kernels, ReLU, 2x2 max pooling;
  - four 3x3 kernels, sigmoid, 2x2 max pooling;
  - a fully connected softmax layer.

  SciPy's `correlate` and `correlate2d` do the convolution sums. I wrote the layer wiring, the max
  pooling and the backward pass myself. It's unfinished; see Known issues.

## Running it

```bash
pipenv install
pipenv run python threeLayer.py      # choose 3 for the full test, or 1 to retrain
pipenv run python gradientCheck.py   # compares backpropagation with numerical gradients
```

Keras downloads Fashion-MNIST the first time you run it. The menus ask which weights file to load: the
`Retrained` files are the 2026 results and the others are the originals. To reproduce a 2026 result,
choose 1 and enter 100 epochs, a learning rate of 0.1, a batch size of 64 and seed 0.

## Fixed since

I found these when I came back to the code in 2026, and fixed each in its own commit. The original
version is tagged `a-level-original`.

- In the fully connected networks, the bias gradient summed over every unit as well as the batch
  (`np.sum(dZ)`), so every bias in a layer got the same update. In the output layer that sum is always
  zero, because each softmax column and each one-hot column add up to 1, so the output biases never
  trained at all. It's now `np.sum(dZ, axis=1, keepdims=True)`. `gradientCheck.py` found it: every
  weight array had a relative error of about 1e-10, and every bias array 0.7 to 1.0.
- The accuracy printed during training came from the last mini-batch, which is 32 images. It now
  covers the whole epoch.
- Training had no seed, so no run could be repeated. `gradientDescent` now takes one.
- Softmax subtracted the largest value in the whole batch rather than in each column. That gives the
  right answer, but if one image's outputs were far larger than another's, the second could come out
  as 0/0. It now uses each column's maximum. This made no difference to any result here.
- Retraining overwrote the original weights. It now saves to the `Retrained` files.
- `mnist.py` had no test evaluation.

## Known issues

- In `convolution.py`:
  - The first layer's ReLU output is overwritten, because pooling is applied to the pre-activation.
  - The max-pooling backward pass copies each gradient to all four cells of its window, instead of
    sending it only to the cell that held the maximum.
  - The menu at the bottom was copied from `threeLayer.py` and doesn't match the CNN's functions.
- `cnnVariables.json` is a copy of `variables.json`, not CNN weights.
