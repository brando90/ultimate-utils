# ---- Rephrase for scientific writing ----

# Prompt: Re-write list of ideas/points to a professional latex itemize list
Provide me with different re-writes of the scientific list in quotations to a latex itemized list.
Remain professional, concise, scientifically sounding, persuasive yet simple, use the active voice:
```latex
% rewrite list bellow to latex list
Limitation:
- div coeff is an aggregate metric/score. Thus, it hides causal factors. (though we show how to use it to find casual factors with the vocab size, latent space. But those experiments are expensive)
- Task2Vec embeddings might be expensive. It's not simply a forward pass and checks the activations. However, given prev work showing it correlates with ground truth task generative parameters, we conjecture it's better than activations. Plus activations might lead to high distances simply because the models were optimized to have a good decision boundary, so we conjecture they are not as good when seeking distances.
- aggregation function, expectation might be arbitrary. Vendi score suggests something else, but it's expensive and under-explored (uses eigenvalues). Further exploration of summing the total distance and vendi are interesting for future work. We predict it won't make a big difference in analogy with the central limit theorem CLT that Gaussian still converges to Normal distribution given a normalization. We conjecture a behavior. But this requires further work.
- we don't think using models is a limitation, as we previously explained in the discussion section. It allows better embeddings of data/batches. A "model" is always needed. The identity function is still a way to represent the data. 

Implications:
- Given LLMs perform impressively, we conjecture our work suggests via a correlation that this might be one reason for their impressive performance. 
- More diversity means more tasks. Thus, we conjecture that high diversity in the pre-training set means a higher chance of including relevant data for testing. This implies one way to improve performance might be via the collection of more diverse data using div coeff (for a sufficiently large model). More direct experiments for this are left for future work.
- Implication is a quantitative transition from a qualitative to a quantitative way of thinking about diversity. Conceptual moves are impactful.
- given we are using Task2Vec to embed data given a model implies one can in principle use our method for any modality, the collect more diverse data. This implies it can improve every field of machine learning. Future work.
```
It should be of top quality for a NeurIPS NIPs ICML ICLR AAAI CVPR machine learning publication
(do not change citations e.g. \citep{...} \cite{...}, urls or names). 
Also, do not change any part that is already excellent. 
Do not sound exaggerated or pompous. 
Keep it concise scientific, use the active voice. 
If it's a list of bullet points, re-write it top quality latex itemized list. 
Use scientific language where appriopriate e.g., using words like conjecture, hypothesis, implication, demonstrate, etc.
If scientific language is used keep it as is.
Do not change the name of my scientific method or techniques.
Provide 3 re-phrased options:

## Great outputs I choose:
TODO

# Prompt: multiple sentences (of one idea) to multiple re-phrasings
"""
I have a couple of different phrasing for a sentence in a scientific paper I am writing in the quotes bellow.
I would like more re-phrasing options:
```text
Option 1: 
We propose to start discussion of data diversity as a metric of data quality, measuring task coverage.

Option 2:
We pioneer discussion on quantitative metrics of data quality by examining task coverage via the concrete measure of the diversity coefficient.
```
It should be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication
(do not change citations e.g. \citep{...}, urls or names).
Also, do not change any part that is already excellent.
Do not sound exaggerated or pompous.
Keep it concise, scientific, use the active voice.
Re-write it amd provide 20 better re-phrased options:

## Great outputs I choose:
TODO




# -- Notes --
We need to re-write this sentence;
```markdown
Therefore, vast amounts of effort have been invested in understanding neural scaling laws 
-- empirical findings that deep artificial networks exhibit power law scaling in performance metrics as a function of 
the \textit{size} of the pre-training dataset, model given a compute budget 
\citep{schaeffer2023emergent, hestness2017deep,rosenfeld2019constructive,henighan2020scaling,kaplan2020scaling,gordon2021data,hernandez2021scaling,jones2021scaling,zhai2022scaling,hoffmann2022training, clark2022unified, neumann2022scaling}.
```
```markdown
Provide me with different re-writes of the scientific text in quotations.
Remain professional, concise, scientifically sounding, persuasive yet simple, use the active voice:
```latex
% rewrite/rephrase bellow
Therefore, vast amounts of effort have been invested in understanding neural scaling laws 
-- empirical findings that deep artificial networks exhibit power law scaling in performance metrics as a function of 
the \textit{size} of the pre-training dataset, model given a compute budget 
\citep{schaeffer2023emergent, hestness2017deep,rosenfeld2019constructive,henighan2020scaling,kaplan2020scaling,gordon2021data,hernandez2021scaling,jones2021scaling,zhai2022scaling,hoffmann2022training, clark2022unified, neumann2022scaling}.
```
It should be of top quality for a NeurIPS NIPs ICML ICLR AAAI CVPR machine learning publication
(do not change citations e.g. \citep{...} \cite{...}, urls or names). 
Also, do not change any part that is already excellent. 
Do not sound exaggerated or pompous. 
Keep it concise scientific, use the active voice. 
Provide 20 re-phrased options: 
```