# ---- Streamlining tweeting for research ----




# Prompt1: Help me write a few top quality short tweets (150 chars) to summarize my research and to later share on Tweeter
PDF: https://arxiv.org/pdf/2306.13840.pdf
Code repo: https://github.com/alycialee/beyond-scale-language-data-diversity/tree/main/diversity#quick-start
Help me write a few top quality short tweets (150 chars) to summarize my research and to later share on Tweeter using 
the given PDF. 
This is an example of a top quality set of tweets:
```markdown
# -- example 1: https://twitter.com/RylanSchaeffer/status/1683654126977314816?s=20
# - tweet 1 example 2
Excited to share our #ICML #ICML2023 paper **Emergence of Sparse Representations from Noise** led by
@TrentonBricken
and supervised by Bruno Olshausen, and @gkreiman!
1/8

# -- example 2: https://twitter.com/RylanSchaeffer/status/1653141214110322688?s=20
# - tweet 1 example 2
Excited to share our #ICML #ICML2023 workshop paper

**Are Emergent Abilities of Large Language Models a Mirage?**

Joint w/ @sanmikoyejo & @BrandoHablando

https://arxiv.org/abs/2304.15004
1/12
[retweeted from Aran Komatsuzaki, account that tweets a lot about others work so tweet has a nice image of our paper]

# - tweet 2 example 2
Recent work claims LLMs display emergent abilities. What makes emergent abilities intriguing is two-fold: 1) their sharpness, transitioning seemingly instantaneously from not present to present & 2) their unpredictability, appearing at seemingly unforeseeable model scales.

2/12
[image embedded from the figure of the paper showing a sudden jump]

# - tweet 3 example 2
We investigate whether these phenomena are truly fundamental to LLMs or might be artifacts of metric choice. Answer: for many claimed abilities, probably artifacts. How? We show, on fixed model outputs, how to choose a metric to create an emergent ability or ablate it!

3/12
[image showing the figure thresholding function is a typical emergent metric]

# 0 tweet example 3
If a model family’s loss falls w/ scale (A), then per-token error rate asymptotes towards 1 (B). If one chooses metric that discontinuously or nonlinearly scales the error rate, emergence (C, D)! But if one doesn’t, no emergence (E, F)! All scored on the same model outputs

4/12
[image showing the figure from the paper showing the ablations of emergence]

...
```
Generate me a good number of good tweets (8-13) summarizing the attached pdf paper.
Include one tweet about a limitation at the end. 
Include the code repo as your second tweet, so that people can easily the work in my work in the PDF:

```markdown
# tweet 1
Excited to share our #ICML #ICML2023 workshop paper 
**Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data**
Co-lead with @_alycialee joint w/ @sanmikoyejo

https://arxiv.org/abs/2306.13840

#NLP #ML #LLM 

1/13

# tweet 2
We release our code with a quick-start to help do hands-on evaluations of dataset diversities. 
Stress test our metric & help us ground data diversity in a quantitative dialogue!

https://github.com/alycialee/beyond-scale-language-data-diversity/tree/main/diversity#quick-start

thanks @_alycialee!

#NLP #ML #LLM #code

2/13

# tweet 3
We introduce the Task2Vec diversity coefficient to measure the diversity of natural language datasets.
We hope this inspires even more thoughtful creations of datasets -- that go beyond scale alone.

#NLP #ML

3/13

[img tweet3 - div eqn]

# tweet4
Our analysis shows open LLMs like GPT-J 6B are pre-trained on datasets with 3-5x more diversity than well motivated 
lower bound (~use same token) 
and 
about half of an upper bound on diversity (~uniform random tokens). 

#NLP #ML

4/13

[img tweet 4 - table 1 with divs]

# tweet5
Combining heterogeneous datasets increases diversity according to the Task2Vec coefficient, matching intuition. 

#NLP #ML

5/13

[img tweet 5 - table 1 focus on concat]

# tweet6
The diversity coefficient correlates with intuitive properties of diversity:
1. increase in number of (latent) concepts --> increase in div. coef.
and 
2. increase in vocabulary size --> increase in div. coef. 

on the synthetic Generative IN-Context Learning (GINC) dataset.
   
#NLP #ML

6/13

[img tweet6 - ginc]

# tweet7
Distributions of pairwise batch distances in our work confirm intuitive properties of the div. coeff - 
unrelated datasets have higher diversity when combined compared to their standad alone diversity. 

#NLP #ML

[img tweet 7 - dist properties]

# tweet 8
We provide guidelines for efficiently computing the diversity coefficient in practice -- even when using random probe networks. 
But using random probe networks is left for future work.

#NLP #ML

[img tweet 8 - guidelines and rand net]

# tweet 9
Embedding methods like Task2Vec approximate semantics between texts, going beyond human-constructed concepts. 
We conjecture this is a strength and enables a data-centric approach to measuring diversity.

#NLP #ML

# img tweet 10 - limitation causal facts
Limitation: The diversity coefficient provides an aggregate measure that masks causal factors underlying diversity. 
But we do show that through additional experiments, these factors can partially revealed.

#NLP #ML

# img tweet 11 - ground truth div
Previous work show div. coef. correlates with ground truth diversity on a synthetic Gaussian benchmark strengthening the trust in the metric. 

prev: https://arxiv.org/abs/2208.01545

#NLP #ML

[img tweet 11 - ground truth div]

# img 12 - future work & conjecture
Future work: Given the impressive performance of LLMs and our diversity measures, we conjecture that high diversity of pre-training data enables this impressive performance of LLMs.

#NLP #ML

# img 13 - extra credits
We'd also like to acknowledge @RylanSchaeffer for providing updated scripts to run GINC and wandb integration.

https://github.com/alycialee/beyond-scale-language-data-diversity/tree/main/ginc#acknowledgements

#NLP #ML
```

examples:
- Claude 2.0: https://claude.ai/chat/7a4ce0f6-9869-43f1-a1e5-8683ddcd5e47

# Prompt2: Help me write an advertisement for the time of my research poster
Help me write an advertisement for the time of my research poster.
This is a good example:
```markdown
Visit LeanDojo's poster and talk to us about LLMs for theorem proving!
7/28 Friday, 11:45 AM – 12:15 PM, room 301, 
@ the ICML Knowledge and Logical Reasoning in the Era of Data-driven Learning workshop.
```
This is my current draft:
```markdown

```
Help me write it and give mee feedback on it. 
Don't change facts on it like the time, date, title or authors of the paper or urls:

# Notes



---

https://twitter.com/BrandoHablando/status/1675361610209325057?s=20

https://twitter.com/_akhaliq/status/1673507375515594766?s=20