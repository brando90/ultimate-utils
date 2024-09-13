# Goal
We want to have a DSPy python program that is truly aware of the final task. 
For example,
1. Step 1 (Synthetic Data Gen): translate Python + Docstring code to Lean code + Docstring (Task 1: Python2Lean)
2. Step 2 (Final Task): AutoFormalization of Docstring to Lean (Task 2: Eng2Lean)

The main feature is that Step 1 potentially has it's own evaluation metric that people usually optimize for (e.g., faithfulness of Python to Lean translation on say CE/TFA) but what we really care about is that the model (we fine-tune usually) does the real final task we care about, in this case AutoFormalization for Lean 4. 

The main challenges are:
1. DSPy doesn't have a Fine-tune module, as of this writing it has a fine-tune Optimzier (Telepromter), which it's main purpose is to optimize/compile a DSPy program/module.

In our case we want to be able to extract from the DSPy compiled program (for any new model but fixed after compilation) the Synthetic Data Generation portion e.g.,

```bash
context --> [synthetic data gen] --> synth data --> [train/ft dspy module] --> maths solution to a problem
```

and we want to create a synthetic data gen module that is aware that it should be creating an optimized module for data gen **that is aware that the final metric and step in the pipeline is fine-tuning a model**. 
So the output after compilation is:
```bash
context --> [synthetic data gen] --> synth data
```
For offline creation of synthetic data. 

The main problems we solve are:
1. We directly minimize the true metric of interest (AutoFormalization of Lean 4) for the real task at the end instead of a proxy task (Quality of Pythont to Lean 4 translation).
2. We an plug in a new model each time that has been pre-trained on different data, so DSPy might fine an optimal prompt for that specific model's drawbacks. So we generate new Synth Data Pipelines for **any** model.  

Then we can create a large offline data set targeted for the specific model and the real task of interest. 

ref: 
    - https://chatgpt.com/share/66e4bb8c-56c4-8001-b9e8-9155e98b3cb3