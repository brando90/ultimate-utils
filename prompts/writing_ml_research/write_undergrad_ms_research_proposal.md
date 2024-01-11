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

# Example 5
Brando Miranda, contact email: brando9@stanford.edu
AI/ML
Aut_win_spr, 2023-2024 Academic Year
Course credit
Up to 5 students
Project Description
Autoformalization is crucial for converting informal statements, typically in natural language, into formal, verifiable statements, such as those in Python, Lean, Coq, or Isabelle. This project aims to develop a reliable evaluation benchmark to measure the capability of machine learning models in translating natural language specifications to formally verifiable specifications using the Interactive Theorem Prover, Lean. The task to evaluate, termed as AF (AutoFormalization), will assess whether a model can create a formalization from an informal statement that is semantically equivalent to the target formalization. The project will involve creating a benchmark with ground truth pairs of informal and formal statements, developing an equivalence function as a score/loss function, and coding a full pipeline that runs evaluations, integrating an AF LLM model, equivalence score/loss function, and LeanDojo. This endeavor is motivated by the aspiration to build an automated mathematician capable of unlocking the vast knowledge encapsulated in mathematical textbooks written in natural language, contributing to advancements in mathematics, scientific discovery, and AI safety.

Recommended Background
Please share your background when reaching out.
Interested candidates are encouraged to share their background when reaching out. A strong foundation in Python is essential, and knowledge in theorem proving using Lean, Coq, or Isabelle is preferred but not mandatory. A passion or intense curiosity about mathematics, formalization/verification of mathematics, AI safety/alignment, or software verification & verified program synthesis would be ideal.

Prerequisites / Preparation
Participants will be expected to make direct contributions to the project and should be comfortable coding in Python. Familiarity with theorem proving and a keen interest in mathematics or software verification would be advantageous.

```
Example 4 & 5 has fields that have to be included when you convert my idea to a proposal. 
This is the idea to convert to a proposal like Example 4 (Brando Miranda will be the supervisor):
```text
# My project
Title
Can Formal Mathemtics Improve Informal Mathematical Reasoning in Large Language Models (LLMs)?

Brando Miranda: brando9@stanford.edu
AI/ML
Aut_win_spr, 2023-2024 Academic Year
Course credit
Up to 5 students
Project description
- idea: Interactive Theorem Provers (ITPs) like Lean, Coq, Isabelle, have libraries of verified proofs for mathematics and programming languages. Therefore, one can mine the step by step proofs from these formal languages.
- Can these formal verified proofs that we mine be used to improve LLMs abiility to reasoning "informally" (i.e. reasoning/mathematics in natural language e.g., English)
- e.g., can we used these mined proofs as is and fine-tune a SOTA model like Mistral to improve the informal reasoning abilities of large language models?
- e.g., 
can we used these mined proofs, "informalize them" (i.e. translate them to informal/natural language mathematics) and fine-tune a SOTA model like Mistral to 
to improve the informal reasoning abilities of large language models on important data set's like the MATH data set?
- the MATH data set is a standard and an important evaluation data set/benchmark for the evaluation of the reasoning capabilities of LLMs.

Recommended Background
Please share your background when reaching out.
Interested candidates are encouraged to share their background when reaching out.
A strong foundation in Python is essential. 
Knowledge in Machine Learning (ML) is recommended.
An interest in LLMs is also recommended. 

Prerequisites / Preparation
Participants will be expected to make direct contributions to the project and should be comfortable coding in Python.
The believe that the role of the data is paramount/most essential is a plus. Data centric ML. 
```
Convert my project into a research proposal with the given specifications, make sure it has the fields as in example 4 & 5. 
Don't forget have at least 1 sentence about the motivation at the end, written in an inspiring yet not exagerated manner. The motivation must be in the project description section.  
It is ok the improve the title given the project Brando gave. 
Make sure to include every bullet point in the project proposal.   
The main content of your propsal should come from the `# My Project` section.
Write the proposal: 
