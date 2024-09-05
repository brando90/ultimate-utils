"""
Goal: replace softmax with somethig better -- that solves the problem with them, if the problem even exists!

First, identify if softmax is even the source of the problem for transfomers.
- Q1: analytically 1D, matrix
- Q2: experiments with grad. How does the gradients behave? Is the step function thing I've see the issue?
- Q3: train a model with sf vs replaced based on what you predicted would help?

ideas:
    - plots with input to sf vs sf input with normalized stuff
    - play around with your custom ViT-22B transformer with different sf's and see what happens to gradients
    - plot paths of gradients for a few layers of a transformer?
"""
#%%