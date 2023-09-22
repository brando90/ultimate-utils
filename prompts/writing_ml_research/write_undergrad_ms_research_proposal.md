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
Example 4 has fields that have to be included when you convert my idea to a proposal. 
This is the idea to convert to a proposal like Example 4 (Brando Miranda will be the supervisor):
```text
# My project
Title
Static benchmark evaluation for AutoFormalization (AF) using Lean Dojo with a Theorem Prover for Equivalences

Brando Miranda: brando9@stanford.edu
AI/ML
Aut_win_spr, 2023-2024 Academic Year
Course credit
Up to 5 students

Autoformalization is the task of converting informal statements (e.g., Natural language) to formal (verifiable) statements (.e.g, Python, Lean, Coq, Isabelle).
The idea is to create a eval benchmark where we can measure reliably if a model is capable of translating natural language specificiations to formally verifiable specificiations (in the ITP Lean).
Thus the task is:

> Task = AF (AutoFormalization) =: can a ml model create a formalization (from an informal statement) that is (formally) semantically equivalent to the target formalization? `AF == i_stmt -> f_stmt`

The main components we will need are:
1. A benchmark with ground truth pairs of informal statements to formal statements (specifying Task AF via examples) see my current public hf data set [debug1](https://huggingface.co/datasets/brando/debug1_af) or [ProofNet](https://huggingface.co/datasets/hoskinson-center/proofnet)
2. An **equivalence** function to be used as a score/loss function. It tells us (ideally) **perfectly** if a traslated/autoformalize informal statement is equivalent to the target formal statement.
3. Full pipeline code that runs eval given:
   - a. (AF) LLM model
   - b. Equivalence score/loss function with a prover capable of proving true equivalences e.g., `fs1 === fs2 ? | Prover, ITP`
   - c. An ITP (Interactive Theorem Prover, Lean). In this case LeanDojo.

Motivation: The dream is to build an automated mathematician that is powerful enough to automate mathematics, sciencific discovery, and AI safety with an automated mathematician. I conjecture formal maths is the only way to create safe AGI because safety requires a "for all" quantifier saying "there is no way AGI will kill humanity". That type of statement are impossible to guarantee empirically and only a mathematical formal proof can guarantee it. Hence why I think building an automated mathematician is the only way for safe AGI.
With this in mind, there is tremendous amount of information that hasn't been unlocked in mathematical textbooks written in natural language (informal langauge), hence the paramount importance of autoformalization in line with the LLM's success trained at scale. 
refs:
AF: https://arxiv.org/abs/2205.12615
AF video: https://youtu.be/_pqJYnQua58?si=jVliUTqqXTjpeods&t=1
ProofNet: https://arxiv.org/abs/2302.12433
ProoNet data set: https://huggingface.co/datasets/hoskinson-center/proofnet
ProofNet github: https://github.com/zhangir-azerbayev/ProofNet
LeanDojo: https://github.com/lean-dojo
Recommended Background
Please share your background when reaching out.
Prerequisites / Preparation
You will be expected to make direct contributions to the project. 
Need to be comfortable coding in python.
Knowledge in Lean/Coq/Isabelle (Theorem Proving) is prefered but not required.
Ideally, you are passionate/intensely curious about mathematics or/and the formalization/verification of mathematics or AI safety/alighment.
Or software verification & verified program synthesis.
```
into a research proposal with the given specifications, make sure it has the fields as in example 4. 
It is ok the improve the title given the project Brando gave:
