# Understanding In-Context Learning using Simple Models

## In-context learning of function classes

- really a paradigm shift. The model was only trained to predict the next token. Why on earth would it be able 
to translate. Or do other crazy things (not in the data)

- it's been tested that it's not memorizing (the data translation with weird symbols)

- defining in-context learning
- in-context learning a **function class**

- double descent, what is the meaning of it and waht is it?

- we really need to study model size contribution Alycia.

- learning is done by giving new linear functions it has NOT seen

### Summary

- really evaluate transformer on sobustness to out-of distributions


## Part 2: Bayseian

Pr[y_qry | x1,y1 ..., xk,yk, x_qry]

rather tha fine tuning with gradient descent

questions:

1. ...
2. how does ICL from training

- main challenge with distribution shifts
- pre-training distribution != prompting distribution

take aways

- theory: Pr[y_qry | x1,y1 ..., xk,yk, x_qry]
- if prompt distribution is close to training distribution this is good
- use neutral delimiters, that don't increase the chance of confusing the LLMs in the ICL

- pre-training val los doesn't improve as model size improves but meta test accuracy improves

- **0 shot is better than 1-shot?! **
- random examples in prompts
- don't damage performance of ICL so much ... meaning prompting ICL is not the same as SL
- bayesian sort of explain when the y's are ranodm are ignored but we do have x's so that helps!

- **ppl in practice do instruction tuning so that FMs/LLMs work in practice**

# Questions

- Q1: do your LSTMs have attention?
- Q2: does Percy think any of this has to do with properties of the data?
- Q3: 0 shot is better than 1-shot? 
- Q4: double descent
- Q5: chain-of-thought, scrap the idea of a task?!
- Q5: knowledge distillation - what is this again?
- Q6: can ICL do visual
- Q7: instruction tuning


