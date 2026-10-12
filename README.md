# Neural networks from scratch

I wrote these during my A-levels (2024-25) to learn how neural networks train. Everything is NumPy, plus
SciPy for the CNN's convolution sums, with no ML framework. Keras is only used to download the datasets.

I started with Harrison Kinsley's (sentdex) [Neural Networks from Scratch](https://nnfs.io) videos, and
`neuron.py`, `objects.py` and `dataset.py` come from those. The series wasn't finished, so I moved to
Samson Zhang's video,
[Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU). `mnist.py`
follows his code closely, including a bias-gradient bug I've since fixed (see Fixed since). The later
scripts build on it: mini-batches and the menu in `mnistBatches.py`, a second hidden layer in
`threeLayer.py`, and the CNN in `convolution.py`.

## Results

These come from retraining in 2026, after the fixes listed under Fixed since. Each network was trained
once with each of the seeds 0, 1 and 2, and accuracy is measured on all 10,000 Fashion-MNIST test images.

| Script | Network | Mean test accuracy | Range over seeds | Saved weights (seed 0) |
|---|---|---|---|---|
| `mnistBatches.py` | 784-10-10: one hidden layer of 10 ReLU units | 84.4% | 83.7 to 84.8% | `variablesRetrained.json`, 83.7% |
| `mnistBatches.py`, `hiddenUnits=20` | 784-20-10: one hidden layer of 20 ReLU units | 85.4% | 85.2 to 85.8% | not saved |
| `threeLayer.py` | 784-20-20-10: two hidden layers of 20 ReLU units | 86.1% | 85.7 to 86.4% | `threeVariableRetrained.json`, 86.3% |
| `convolution.py` | CNN: 2 then 4 convolution kernels and a softmax layer, 890 parameters | 83.4% | 83.0 to 84.1% | `cnnVariables.json`, 83.0% |
| `convolution.py`, `kernels1=8, kernels2=16` | CNN: 8 then 16 convolution kernels and a softmax layer, 4,394 parameters | 87.8% | 87.6 to 88.4% | `cnnVariables8then16.json`, 88.4% |

The fully connected networks were trained for 100 epochs with a learning rate of 0.1 and a batch size of
64. I didn't record the settings I used in 2024, so these are a reconstruction: they're the values in a
commented-out training call in the original code (tagged `a-level-original`). Both CNNs were trained for
30 epochs, one image at a time, with a learning rate of 0.01 that's multiplied by 0.97 after each epoch.

I chose seed 0 for the saved weights before training. On my machine, retraining with seed 0 reproduces
each saved weights file exactly. Another machine's maths library may round differently in the last
decimal places.

The original 2024 weights are still here: `variables.json` scores 84.0% and `threeVariable.json` scores
85.4%. Their settings and random starting weights weren't recorded, so they can't be reproduced. There
were no original CNN weights; `cnnVariables.json` used to be a copy of `variables.json`.

### Did fixing the bias gradient change accuracy?

Not measurably. I retrained both fully connected networks with the same seeds and settings, but with the
old bias gradient put back. They averaged 84.8% (784-10-10) and 86.0% (784-20-20-10). For each seed, the
fixed and buggy versions were within 0.7 points of each other, while different seeds of the same version
varied by up to 1.1 points. For the smaller network the buggy version scored slightly higher on all
three seeds, but three seeds can't separate a difference that small from chance.

The biases are a small part of each network (20 of 7,960 parameters in the smaller one, 50 of 16,330 in
the larger), and the weights were always trained correctly, so they made up for it. The accuracy never
showed the bug; a gradient check did.

### Did fixing the CNN change accuracy?

Yes. I retrained the 2 then 4 CNN with the same seeds and settings, but with the two max-pooling bugs
put back: pooling the first layer's values from before ReLU, and copying each gradient to all four cells
of its window. That version averaged 73.9% (70.0 to 77.5%), against 83.4% for the fixed one. The fixed
version was ahead on every seed, by 6.7 to 13.0 points, and the two ranges don't overlap. The other two
CNN fixes, float64 images and a working menu, are in both versions.

The version with the bugs also trained erratically. From one epoch to the next its training accuracy
went up and down by as much as 4 points, and it never settled. The fixed version's rose steadily and
levelled off at 83.7 to 85.1%.

A shorter run beforehand (50 rounds of 3,000 images, two and a half epochs' worth) gave 79.7% fixed
against 74.7% with the bugs. Training on twelve times as many images raised the fixed version by 3.7
points and left the version with the bugs no better.

### Does the second hidden layer help?

784-20-10 and 784-20-20-10 differ only in the second hidden layer, so comparing them isolates depth.
The second layer added 0.7 points on average (85.4% to 86.1%) for only 420 extra parameters. The two
ranges overlap slightly (85.2 to 85.8% against 85.7 to 86.4%), so with three seeds each this is
suggestive rather than settled. For comparison, widening the single hidden layer from 10 to 20 units
added 1.0 point (84.4% to 85.4%) and doubled the number of parameters.

The 784-20-10 runs call `gradientDescent` in `mnistBatches.py` with `hiddenUnits=20`; the menu only
trains the 10-unit version.

### Does a bigger CNN help?

Yes. The 2 then 4 CNN averaged 83.4%, below all three fully connected networks, even though
convolutional networks usually do better on images. It's tiny: 890 parameters, against 7,960 in the
smallest fully connected network. Its training accuracy levelled off at 83.7 to 85.1%, only a little
above its test accuracy, so it wasn't memorising the training images; it looked too small to fit them
any better. To test that, I trained it with 8 then 16 kernels and kept everything else the same: 30
epochs, the same learning rate and the same seeds.

It went from 83.4% to 87.8% (87.6 to 88.4%), and the two ranges don't overlap. That's also above the
best fully connected network (86.1%, 85.7 to 86.4%), with about a quarter of its parameters (4,394
against 16,330). Its training accuracy reached 90.6 to 90.9%, so the gap between training and test
accuracy grew from under a point to between 2.5 and 3.1 points: it's starting to learn details of the
training images that don't carry over to new ones. Test accuracy still went up, so the extra kernels
were worth it.

Before training it, I ran the gradient check on a CNN with 4 then 8 kernels, and it agreed with
backpropagation to about 1e-10, the same as the default size. The 8 then 16 runs call `gradientDescent`
in `convolution.py` with `kernels1=8, kernels2=16`; the menu only trains the default size. Each run took
about five hours on my laptop, so I've only tried this one bigger size.

The seed 0 weights are in `cnnVariables8then16.json`. The menu only loads `cnnVariables.json`, so to
test them, pass the file name to `loadVariables`, then pass what it returns to `test` with `test_X` and
`test_y`. I haven't repeated the five-hour run to confirm that retraining reproduces this file exactly,
as I did for the others.

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
- `convolution.py`: a small convolutional network, trained one image at a time with a decaying learning
  rate:
  - 9x9 kernels (2 by default), ReLU, 2x2 max pooling;
  - 3x3 kernels (4 by default), sigmoid, 2x2 max pooling;
  - a fully connected softmax layer.

  SciPy's `correlate` and `correlate2d` do the convolution sums. I wrote the layer wiring, the max
  pooling and the backward pass myself. Its menu works like the others, and its weights are in
  `cnnVariables.json`. `gradientDescent` takes `kernels1` and `kernels2` to change the number of
  kernels in each layer, and the 8 then 16 weights are in `cnnVariables8then16.json`.
- `gradientCheck.py`: checks the backward pass of `mnistBatches.py`, `threeLayer.py` and
  `convolution.py`. It works out each gradient a second way, by nudging every parameter up and down and
  measuring the change in the loss, and prints the relative error against backpropagation for each
  parameter array. Below about 1e-7 means backpropagation is right; every array currently comes out at
  about 1e-10.

## Running it

```bash
pipenv install
pipenv run python threeLayer.py      # choose 3 for the full test, or 1 to retrain
pipenv run python convolution.py     # the same menu for the CNN
pipenv run python gradientCheck.py   # compares backpropagation with numerical gradients
```

Keras downloads Fashion-MNIST the first time you run it. The fully connected menus ask which weights file
to load: the `Retrained` files are the 2026 results and the others are the originals. To reproduce a
2026 result, choose 1 and enter the recommended settings the menu shows, with seed 0. For the CNN that's
30 rounds of 60,000 images, which takes a little over an hour on my laptop.

## Fixed since

I found these when I came back to the code in 2026, and fixed each in its own commit. The original
version is tagged `a-level-original`.

In the fully connected networks:

- The bias gradient summed over every unit as well as the batch (`np.sum(dZ)`), so every bias in a layer
  got the same update. In the output layer that sum is always zero, because each softmax column and each
  one-hot column add up to 1, so the output biases never trained at all. It's now
  `np.sum(dZ, axis=1, keepdims=True)`. `gradientCheck.py` found it: every weight array had a relative
  error of about 1e-10, and every bias array 0.7 to 1.0.
- The accuracy printed during training came from the last mini-batch, which is 32 images. It now covers
  the whole epoch.
- Training had no seed, so no run could be repeated. `gradientDescent` now takes one.
- Softmax subtracted the largest value in the whole batch rather than in each column. That gives the
  right answer, but if one image's outputs were far larger than another's, the second could come out as
  0/0. It now uses each column's maximum. This made no difference to any result here.
- Retraining overwrote the original weights. It now saves to the `Retrained` files.
- `mnist.py` had no test evaluation.

In the CNN:

- The first layer's ReLU output was thrown away, because pooling was applied to the values before ReLU.
  It now pools the ReLU output.
- The max-pooling backward pass copied each gradient to all four cells of its 2x2 window. Only the cell
  that held the maximum affects the output, so the forward pass now records a mask of where each maximum
  was, and the backward pass sends each gradient only there. The gradient check for the convolution
  weights went from about 0.6 to about 1e-10.
- The images were float32, so SciPy's FFT convolution ran in single precision. The gradient check showed
  this as a relative error of 5e-8 in the first layer's weights that didn't change with the step size.
  The images are now float64.
- The menu, saving and loading were copied from `threeLayer.py` and didn't work with the CNN. Saving
  also pointed at `threeVariable.json`. They now train the CNN, save it to `cnnVariables.json`, and test
  it on all 10,000 test images rather than the first 1,000.
- The backward pass assumed exactly two first-layer kernels (`np.zeros((10, 10, 2))`), so changing the
  number of kernels crashed it. It now takes the number from the weights.

I also removed commented-out code, corrected comments that were wrong (one called the softmax and
cross-entropy gradient "the loss"), fixed spellings in names, and renamed variables that hid Python
built-ins. None of that changed any result.

## Limitations

- The CNN trains one image at a time in Python loops, so it's slow: an epoch takes about two and a
  quarter minutes on my laptop with 2 then 4 kernels, and about 11 minutes with 8 then 16.
- Each result comes from three seeds. That's enough to see the effect of the CNN fixes and of the bigger
  CNN, but not to separate differences of half a point or so from chance.
