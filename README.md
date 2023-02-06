# ML trial room

This is a repository for testing out different ML models before actually using them in useful code. This is a place where I learn to use stuff. The folder [mnist](mnist) has models which I have tried training on the mnist dataset, like:
- [an MLP](mnist/mnist_torch.py)
- [the same MLP implemented in Pytorch Lightning](mnist/mnist_pl.py)
- [Perceiver IO](mnist/mnist_perceiver_textbook.py) copied from [here](https://github.com/krasserm/perceiver-io)
- [a Perceiver IO adapted from the earlier version, written in Pytorch Lightning](mnist/mnist_perceiver_pl.py)
- [the same written in plain pytorch](mnist/mnist_perceiver.py)