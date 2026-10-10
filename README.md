# Neural networks from scratch

I wrote these during my A-levels (2024-25) to learn how neural networks train. Everything is NumPy, with
no ML framework. Keras is only used to download the datasets.

I started with Harrison Kinsley's (sentdex) [Neural Networks from Scratch](https://nnfs.io) videos, and
`neuron.py`, `objects.py` and `dataset.py` come from those. The series wasn't finished, so I moved to
Samson Zhang's video,
[Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU). `mnist.py`
follows his code closely, including the bias-gradient bug listed under Known issues. The later scripts
build on it: mini-batches and the menu in `mnistBatches.py`, a second hidden layer in `threeLayer.py`,
and the CNN in `convolution.py`.

## Results

| Script | Dataset | Network | Test accuracy |
|---|---|---|---|
| `mnistBatches.py` | Fashion-MNIST | 784-10-10: one hidden layer of 10 ReLU units | 84.0% |
| `threeLayer.py` | Fashion-MNIST | 784-20-20-10: two hidden layers of 20 ReLU units | 85.4% |

Both were measured on all 10,000 Fashion-MNIST test images, using the saved weights in this repo. Both
were trained with shuffled mini-batch gradient descent.

The two networks differ in width as well as depth, so the 1.4-point gain can't be put down to the extra
layer alone.

## What's in here

- `neuron.py`, `objects.py`, `dataset.py`: my first experiments, following the sentdex videos. A layer
  worked out by hand, then dense layers, ReLU, softmax and cross-entropy loss written as classes and
  tried on a synthetic spiral dataset.
- `mnist.py`: one hidden layer on MNIST digits with full-batch gradient descent, following Samson
  Zhang's video. It prints training accuracy; there's no test evaluation.
- `mnistBatches.py`: one hidden layer on Fashion-MNIST with shuffled mini-batches. An interactive menu
  lets you retrain, test a single image, or run the whole test set. Weights are in `variables.json`.
- `threeLayer.py`: the same with two hidden layers. Weights are in `threeVariable.json`.
- `convolution.py`: a small convolutional network, trained one image at a time with a decaying learning
  rate:
  - two 9x9 kernels, ReLU, 2x2 max pooling;
  - four 3x3 kernels, sigmoid, 2x2 max pooling;
  - a fully connected softmax layer.

  SciPy's `correlate` and `correlate2d` do the convolution sums. I wrote the layer wiring, the max
  pooling and the backward pass myself. It's unfinished; see below.

## Running it

```bash
pipenv install
pipenv run python threeLayer.py    # choose 3 for the full test, or 1 to retrain
```

Keras downloads Fashion-MNIST the first time you run it.

## Known issues

I found these when I came back to the code in 2026. They're being fixed in separate commits, and the
original version is tagged `a-level-original`. The results above come from that original version.

- In the fully connected networks, the bias gradient sums over every unit as well as the batch
  (`np.sum(dZ)`). So every bias in a layer gets the same update. It should be
  `np.sum(dZ, axis=1, keepdims=True)`.
- In `convolution.py`:
  - The first layer's ReLU output is overwritten, because pooling is applied to the pre-activation.
  - The max-pooling backward pass copies each gradient to all four cells of its window, instead of
    sending it only to the cell that held the maximum.
  - The menu at the bottom was copied from `threeLayer.py` and doesn't match the CNN's functions.
- `cnnVariables.json` is a copy of `variables.json`, not CNN weights.
- The accuracy printed during training comes from the last mini-batch only.
