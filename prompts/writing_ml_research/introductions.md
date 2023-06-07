# ---- Write Introduction Research Paper ----
references: 
    - (fantastic discussion section!): Are Emergent Abilities of Large Language Models a Mirage? https://arxiv.org/abs/2304.15004

[//]: # (    - MAML: https://arxiv.org/abs/1703.03400)
[//]: # (    - Hashimoto: http://proceedings.mlr.press/v139/hashimoto21a/hashimoto21a.pdf )

# Suggestion to writing your example to improve:

# Prompt:
Write a top quality introduction for a NeurIPS NIPS ICML ICLR AAAI CVPR machine learning publication from my notes, 
top quality examples, and instructions provided.
Instructions to write great introduction section:
```markdown
Instructions to write a top quality Introduction section for Machine Learning papers:
1.

One line summary for top quality introduction section: The introduction (1 page, 100 readers); 
i. Describe the problem ii. State your contributions iii. And after briefly implication/impact, why it matters.

Other useful points to keep in mind in the writing of the introduction:
- 
```

Here is an example of a top quality introduction section:
```latex
% top example 1
\section{Introduction}

```

Here is my sample introduction section that needs rewriting and improvement (perhaps as a set of bullet points or informal notes). 
Make it top quality for a NeurIPS NIPS ICML ICLR machine learning publication as the instructions and examples require:
```latex
\section{Introduction}

```
My improved introduction section be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication.
Do not change citations e.g. \citep{...}, urls. 
Do not copy the facts but the style of the top abstract example I provided. 
Do not change the facts in my example. 
Do not change any part that is already excellent. 
Do not sound exaggerated or pompous. 
Keep it concise, scientific, direct, use the active voice. 
Follow the instructions conditioned by the example provided above. 
The introduction section should be similar in length to the top example i.e. 1, 2 or 3 paragraphs. 
Provide 3 re-phrased options: 


# -- Notes --

```markdown
What are the best practices or advice for writing a top quality machine learning introduction section for a research paper
on the caliber of NeurIPS, NIPS, ICML, ICLR, CVPR, AAAI? 

Outline the requirements as a list or instructions need to write a good research introduction:
Writing an introduction for a machine learning research paper is a crucial task that sets the tone for the entire paper. It needs to be precise, engaging, and thorough to catch the reader's interest. Here's a list of best practices for crafting a high-quality introduction for a research paper:

Identify the Problem: The introduction should begin with a clear statement of the problem or issue that the research is addressing. This could be a gap in current understanding, a challenge in machine learning applications, or an unexplored research area.

Contextualize the Problem: Once the problem is stated, provide the reader with some context. Describe the current state of research in this area, recent advances, and remaining challenges. This context setting should demonstrate your understanding of the field and the significance of the problem.

Motivate Your Work: Motivate the need for your research. Why is it necessary to solve this problem? How will your research contribute to the existing knowledge in this field? In the case of machine learning, this could relate to improvements in accuracy, efficiency, or new capabilities.

State the Objectives and Contributions: Clearly state the objective of your research. What are you hoping to achieve? Along with this, highlight the main contributions of your paper. What new methods, techniques, or insights does your paper offer?

Outline Your Approach: Provide a brief overview of the approach or methodology you've used in your research. You don't need to go into detail here, as this will be covered in the main body of your paper.

Discuss the Structure of Your Paper: Briefly outline what each section of your paper will cover. This gives the reader a roadmap of what to expect.

Write in Clear, Concise Language: Avoid jargon as much as possible and make sure your introduction is easy to understand. Remember, some readers might be new to the specific area of your research, so clarity is key.

Cite Relevant Literature: Be sure to reference key papers in your field. This shows you have a thorough understanding of the topic and places your work in the context of existing research.

Avoid Overselling: While it's important to highlight the significance of your work, avoid hyperbole and unsupported claims. Be realistic about the implications and potential impact of your research.

Revise and Edit: Last but not least, ensure you revise and edit your introduction carefully. It's the first part of your paper that reviewers will read, so it needs to be polished and error-free.

Remember that the introduction is just a preview of your research - you will delve into more detail in the subsequent sections of your paper. Following these guidelines will help you create an introduction that is compelling, comprehensive, and sets up the rest of your paper effectively.
```

Simon Jones: https://www.microsoft.com/en-us/research/uploads/prod/2016/07/How-to-write-a-great-research-paper.pdf
The introduction (1 page, 100 readers)
- Describe the problem
- State your contributions
- And after briefly implication/impact, why it matters (my addition)
...and that is all
ONE PAGE!
```markdown

```
todo: I feel this is related to implication too, so that ppl know how it affects them & why it matters. 

My paper, I think I like my introduction tbh: https://www.overleaf.com/project/647aa73fd4e9184c9e9ad43d
```markdown
\section{Introduction}
Current trends in pre-training Large Language Models (LLMs) tend to concentrate on model and dataset size scaling \cite{chowdhery2022palm, nostalgebraist2022chinchilla, gpt4, google2023palm2}.
Therefore, vast amounts of effort have been invested in understanding neural scaling laws -- empirical findings that deep networks exhibit power law scaling in test metrics as a function of the \textit{size} of the pre-training dataset, model or compute \citep{hestness2017deep,rosenfeld2019constructive,henighan2020scaling,kaplan2020scaling,gordon2021data,hernandez2021scaling,jones2021scaling,zhai2022scaling,hoffmann2022training, clark2022unified, neumann2022scaling}.
However, the effectiveness of these systems fundamentally relies on the quality \cite{longpre2023pretrainer} and coverage of the pre-training data \cite{tatsu, david2010impossibility}.
Unfortunately, data quality and coverage \cite{david2010impossibility} are often overlooked or discussed in vague and imprecise ways \cite{longpre2023pretrainer}, we propose to ground the discussion of data quality through the diversity coefficient \cite{curse_low_div}, a data coverage metric that moves beyond scale alone.
We extend the diversity coefficient to formally quantify data diversity of publicly available datasets and discover that LLMs are pre-trained on formally diverse data.
We demonstrate the diversity coefficient is \textit{high} for these pre-training datasets by comparing their formal diversity to the non-vacuous conceptually well-motivated lower and upper bounds of the diversity coefficient. 
In addition, to instill confidence in the usage of the diversity coefficient, we assess the interpretability of the coefficient as it relates to intuitive and expected properties of such a diversity metric.
Concretely, we demonstrate:
\begin{enumerate}
    \item The diversity coefficient increases as one concatenates more pre-training datasets of different sources.
%[alycia's wisker plots of single vs concts of two datasets]
    \item We show the task embedding distances used in the diversity coefficient groups in a meaningful way, reflecting the conceptual and semantic information humans expect.
%[pile histogram and MIO vision data set]
    \item Using the Generative IN-Context Learning (GINC) \cite{ginc} dataset, we show that as the number of latent concepts\footnote{Latent concepts represent document-level features such as semantics, structure, and style \cite{ginc}.} increases the diversity coefficient increases.
%[correlation plot div vs latent concept, and new one with lots of line, note we have to decide if we need that many plots to make our point, we can reference additional evidence/plots in the appendix]
    \item We show that a larger, more diverse vocabulary leads to a higher diversity coefficient in the Generative IN-Context Learning (GINC) \cite{ginc} dataset.
%[correlation plot div vs vocab, and new one with lots of lines]
%In addition, we recap and emphasizes that the diversity coefficient has been shown to correlate with the ground truth diversity of synthetic datasets when available.
\end{enumerate}
%In addition, we recap and emphasizes that the diversity coefficient has been shown to correlate with the ground truth diversity of synthetic datasets when available.
Our key \textbf{contributions} are:
\begin{enumerate}
    \item A paradigm shift beyond dataset scale to a data-centric machine learning perspective through a formal data quality metric -- the diversity coefficient.
    \item We advance discussions on data quality by measuring an aspect of quality -- data diversity -- using the diversity coefficient.
    \item We further validate the diversity coefficient by demonstrating its interpretability and correlation with intuitive diversity properties aligned with human intuitions, 
    e.g., the coefficient increases as more datasets are concatenated, the number of latent concepts increases, and a richer vocabulary is used.
    \item We formally demonstrate the high diversity of public datasets for LLM pre-training is \textit{high} using well-motivated lower and upper bounds.
    \item Lastly, for ease of usage of our method, we also study properties of different parameters for computing the formal diversity and therefore provide practitioners with simpler ways to evaluate the diversity coefficient.
\end{enumerate}
Therefore, we conclude the diversity coefficient is reliable, and conjecture the diversity coefficient can be used to build quality diverse datasets for capable LLMs. 
In doing so, we hope this work inspires more systematic and effective techniques for dataset design beyond simply increasing the number of data points, sequences, or tokens.

%I do like this one but unsure where to put without further thoughts...
% LLMs are pre-trained on vast text corpora from diverse sources, and possess the remarkable abilities to perform downstream tasks not explicitly trained for. 
%We show that large language models (LLMs) are pre-trained on highly diverse data sets, and find that the diversity coefficient correlates with intuitive measures of diversity, such as the number and types of data sources. 
```