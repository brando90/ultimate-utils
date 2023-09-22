# Prompts for writing research proposal for Undergrads and Master students

# Prompt: Write a research proposal from an idea and few shot exampls
Write a CS research proposal that is clear, professional for undergraduate or masters student's (MS) level.
It should be:
- clear
- concrete
- understandable
- motivated
- inspiring
- concise, only 1 paragraph maximum
Here is an example of a few examples of research proposals:
```text
# -- Example 1
Title
Exploring Large Language Models' Capabilities in One-Shot Mathematical Proof Generation with In-Context Learning

Interactive theorem provers (ITPs) enable the representation of mathematical proofs as proof programs. However, these programs are typically not explicitly accessible as code, as they exist as internal representations in ITPs. By utilizing the PyCoq interface, we aim to create a dataset of such proof programs and investigate the capabilities of various Large Language Models (LLMs) like PaLM, LLaMA, GPT family,fine-tuned LLMs, etc. to perform the task of one-shot mathematical proof generation. This can be done for example via In-Context Learning. This reasoning task will be integrated into the HELM benchmark for comparative analysis to compare with a wide range of models. Throughout the project, the student will gain experience in evaluating state-of-the-art language models on a novel dataset and environment, utilizing the HELM benchmark to compare their performance with other models available on HELM on challenging reasoning tasks such as one-shot theorem proving. Given sufficient time, we hope to also assess the performance of our fine-tuned Alpaca/LLaMA model on this novel dataset. 

Key Citations:
HELM: https://crfm.stanford.edu/helm/latest/
PyCoq: https://github.com/brando90/ultimate-pycoq 

Supervisors:
Brando https://brando90.github.io/brandomiranda/prospective-collaborations.html 
Rylan

# Example 2
Title
Unraveling the Significance of Data Quality and Diversity in GPT-4 Era and its Connection to Emergent Capabilities

Project Proposal: In the contemporary realm of GPT-4, the development of large-scale language models (LLMs), also known as foundation models, has predominantly been confined to major industrial laboratories. These have primarily emphasized augmenting data volume, model size, and an ambiguous notion of data diversity enhancement. Consequently, this landscape presents a prime opportunity for academia to advance a promising research trajectory by formulating data quality metrics that elucidate the origins of the remarkable emergent capabilities exhibited by models such as GPT-4 [3]. Our proposal advocates for a methodically investigation of diversity [1] as a data quality metric and its correlation with the in-context learning abilities of LLMs [4].

1. The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence  https://arxiv.org/abs/2208.01545 
2. GPT-4: https://arxiv.org/abs/2303.08774 
3. Sparks of Artificial General Intelligence: Early Experiments with GPT-4. https://arxiv.org/abs/2303.12712  
4. GPT-3, Language Models are Few-Shot Learners: https://arxiv.org/abs/2005.14165  

Supervisors:
Brando https://brando90.github.io/brandomiranda/prospective-collaborations.html 

# Example 3
Pretraining Language Models from Human Preferences
Recent work showed that language models can be pretrained using human preference data [1] in a manner that minimizes misalignment without sacrificing task performance (To experiment, the authors used GPT-2 sized models, which are relatively easily trained on the compute we have available to us). The paper explored several methods for utilizing human preference data and found that only one performed well. However, one of the methods that did not work is called Reward Weighted Regression:
$$
\mathcal{L}_{\mathrm{RWR}}(x)=\sum_{i=1}^{|x|} \log \pi_\theta\left(x^i \mid x^{<i}\right) \exp \left(R\left(x^i\right) / \beta\right)
$$
What is puzzling about this objective is that it is extremely similar to the policy gradient but apparently does not work well in practice. This project will first verify the claimed results, then test whether non-exponentiated rewards have similarly poor performance. If the answer is yes, then this suggests that policy gradient approaches contain some secret sauce that we will work to identify. If the answer is no, we have a new method for learning from human preferences without requiring RL.

Key Citations:
Pretraining Language Models with Human Preferences

Supervisors:
Rylan
Zach?

# Example 4
Burst: A Social Media Design in Layers
Michael Bernstein, contact email: msb@cs.stanford.edu
HCI
Aut_win_spr, 2023-2024 Academic Year
Course credit
Up to 5 students
Project Description
It would be no understatement to claim that social media such as Facebook, TikTok, and X are beset by challenges. And while we have developed many tools to help—secondary accounts such as finstas, moderation tools, AI, deplatforming rules, prosocial nudges—these are all ultimately band-aids that patch these problem rather than addressing the design decisions that cause it. In this project, we ask: can we change some of the fundamental core design decisions of social media to come up with an alternative that’s enjoyable and more pro-social?
The current project direction focuses on creating a For You-style feed of content, where your posts get shared with concentric “rings” of followers, and your content bursts from smaller to larger and larger rings. So, when you first post, your content might only be visible to 10 trusted friends. If they react positively, it might burst out to a group of 100, then 1000, and so on. We are developing this idea into a working React Native application to deploy and test its effect on how comfortable people are sharing content, as well as how effectively it can mitigate anti-social behavior such as harassment.
Recommended Background
Please share your background when reaching out.
Prerequisites / Preparation
Most contributors to this project will need to be comfortable coding in React Native, to make direct contributions to the project. We also have a small capacity for contributors interested in user research or design.
```
now convert this idea:
```text
Autoformalization is the task of converting informal statements (e.g., Natural language) to formal (verifiable) statements (.e.g, Python, Lean, Coq, Isabelle).
Create a benchmark for autoformalization from Lean's Mathlib library.
The library will have human judgements/scores of how good a formalization or informalization is.
We will create some data manually similar to the debug1_af data https://huggingface.co/datasets/brando/debug1_af
i.e., we will do the formalizaiton using GPT4 or Claude, then have us (human experts) label how good the
formalization, informalization is.
Once we get at least 500 examples labeled we can train a reward model
Read the LIMA paper to see how they managed to get such a good model with 1000 examples
Then we will train a reward model, evaluate how good at is/algined with human preferences is
and then use it to label all the paired data (informal statement, formal statment) with two scores for evlauation and training.

refs:
AF: https://arxiv.org/abs/2205.12615
AF video: https://youtu.be/_pqJYnQua58?si=jVliUTqqXTjpeods&t=1
LIMA Less is More For Alignment: https://arxiv.org/abs/2305.11206
```
into a research proposal with the given specifications:
