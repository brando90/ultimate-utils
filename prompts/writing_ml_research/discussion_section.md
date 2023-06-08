# ---- Write Captions for Figures in a Research Paper ----
references: 
    - (fantastic discussion section!): Are Emergent Abilities of Large Language Models a Mirage? https://arxiv.org/abs/2304.15004
    - MAML: https://arxiv.org/abs/1703.03400
    - Hashimoto: http://proceedings.mlr.press/v139/hashimoto21a/hashimoto21a.pdf 

# Suggestion to writing your example to improve:
```markdown
First:
1. Write a 1-2 sentence contribution/conclusion/main result.
2. Write a list of benefits and limitations with arguments in favor and against each point. Optionally, if there is future work to mention after limitation it can be mentioned. 
3. Implication & impact (& future work maybe)
4. The choose the points we want to put in paper from points 2, 3
5. Then write the section well.
Instead of going straight to the dicussion section (randomly)
```

# Prompt:
Write a top quality for a NeurIPS NIPS ICML ICLR AAAI CVPR machine learning publication from my notes, 
top quality examples, and instructions provided.
Instructions to write great discussion section:
```markdown
Instructions to write a top quality discussion section for Machine Learning papers:
1. First (perhaps two/three) sentences: highlight concisely the main results, contribution and conclusion.
2. The discussion section should highlight the benefits, without exaggerating. 
3. The discussion section should discuss the limitations of the work in an honest (positive) way without diminishing your findings. Honest work is more trustworthy and thus more impactful.
4. Discuss the implications of the work and impact to the field of machine learning or relevant subfield. Connection to future work can be mentioned here.

One line summary for top quality discussion section: Discussion; Highlight key results and then proceeds to discusses implication (thus the work's potential impact), benefits, limitations, interpretations, compare with relevant literature, and future work (it can be connected to impact and implications).

Other useful points to keep in mind in the writing of the discussion section:
- Make sure discussion sections prose flows well and logically.
- Use the active voice.
- Be concise, clear, and use precise language. If language is precise then it's more likely to be trustworthy since it's more likely to be objective. 
- Discussion sections should reflect critical thinking and evaluations.
- Although your discussion section discusses limitation, it should still make your work positive, persuasive and impactful.
- good starting sentences:  "We propose..." or "We introduce..." or "Our paper presents..."
- Note, future work can be followed up after a discussion of limitation of the current work.
```

Here is an example of a top quality discussion section:
```latex
% top example 1
\section{Discussion}
Our paper presents an alternative explanation for claimed emergent abilities of large language models. 
For a fixed task and a fixed model family, the researcher can choose a metric to create an emergent ability or choose a metric to ablate an emergent ability. 
Ergo, \textit{emergent abilities may be creations of the researcher's choices, not a fundamental property of the model family on the specific task.} 
We emphasize that nothing in this paper should be interpreted as claiming that large language models \textit{cannot} display emergent abilities; rather, our message is that previously claimed emergent abilities in \cite{brown2020language, ganguli2022predictability,srivastava2022beyond,wei2022emergent} might likely be a mirage induced by researcher analyses.

Our paper has several implications. 
Firstly, a task and a metric are distinct and meaningful choices when constructing a benchmark. 
Secondly, when choosing metric(s), one should consider the metric's effect on the per-token error rate and adapt their measuring process accordingly, 
e.g., if one chooses accuracy, one should make sure to have sufficient data to accurately measure accuracy to avoid the risk of drawing invalid scientific conclusions.
Thirdly, when making claims about capabilities of large models, including proper controls is critical. 
In this particular setting, emergent abilities claims are possibly infected by a failure to control for multiple comparisons. 
In BIG-Bench alone, there are $\geq$ 220 tasks, $\sim 40$ metrics per task, $\sim10$ model families, for a total of 
$\sim 10^6$ task-metric-model family triplets, meaning probability that \textit{no} task-metric-model family triplet 
exhibits an emergent ability by random chance might be small.
Fourthly, scientific progress can be hampered when models and their outputs are not made public for independent scientific investigation.

% top example 2
\section{Discussion}
We introduced a meta-learning method based on learning easily adaptable model parameters through gradient descent. 
Our approach has a number of benefits. 
It is simple and does not introduce any learned parameters for metalearning. 
It can be combined with any model representation
that is amenable to gradient-based training, and any differentiable objective, including classification, regression, and
reinforcement learning. Lastly, since our method merely
produces a weight initialization, adaptation can be performed with any amount of data and any number of gradient steps, though we demonstrate state-of-the-art results
on classification with only one or five examples per class.
We also show that our method can adapt an RL agent using
policy gradients and a very modest amount of experience.

Reusing knowledge from past tasks may be a crucial ingredient in making high-capacity scalable models, such as
deep neural networks, amenable to fast training with small datasets. 
We believe that this work is one step toward a simple and general-purpose meta-learning technique that can
be applied to any problem and any model. 
Further research in this area can make multitask initialization a standard ingredient in deep learning and reinforcement learning.

% top example 3
In this work, we’ve proposed a new approach to predicting the performance of a prediction model as a function of
training data composition that consists of measuring model
accuracies for small n and a range of q and fitting a parametric model 
V (n, q) := −α log(n) + \sum_^m_{i=1}( \sum^K_{k=1}\lambda_{i,k} q_k)^−1.
Our main contribution is to show that for a range of models,
this nonlinear parametric model is a more natural approximation to generalization error than existing linear approximations. 
Empirical results on the Amazon sentiment regression
task show that this approximation is accurate under a range
of conditions, and experiments on neural models suggest
that the method can continue to perform well in more realistic situations where the theory does not necessarily hold.
Our work is a first step in going beyond closed-form estimates of model performance or additivity assumptions. 
It is an open question whether the same approach can scale
to more extreme extrapolation settings or large numbers of
data sources, and we hope to explore this in future work.

%top example 4
```

Here is my sample discussion section that needs rewriting and improvement (perhaps as a set of bullet points or informal notes). 
Make it top quality for a NeurIPS NIPS ICML ICLR machine learning publication as the instructions and examples require:
```latex
\section{Discussion}
- However, it is worth noting that our metric relies on tunable parameters, including the number of batches, batch size, and probe network depth. 
	- These parameters affect the absolute value of the diversity coefficient, necessitating recalibration of the conceptual lower and upper bounds accordingly (see sections \ref{appendix:parameter_selection} and \ref{appendix:div_parameters}).
		- However, this can be resolved by using a consistent choice of hyper parameters. 
		- In addition, while we employ a pre-trained and fine-tuned network as a probe, we are actively exploring alternative approaches, such as using random and/or non-fine-tuned networks (see section \ref{appendix:div_parameters}).
- how does diversity relate to quality/performance?
	- diversity might not help models e.g. just uniform noise has the most diversity (NFT)
	- however, since diversity tries to capture spread, a higher diversity (with a sufficiently large model) likely leads to better performance and therefore your data is of higher quality. 
- diversity hides truly causual stuff, but this was by design to make it more practical usable
	- but you can do abalation experiments like we did
	- but it's more widely useful/applicable
	- no silver bullet
- However, the bounds we propose only apply to sequence data with a symbolic vocabulary.
	- However, using a multimodal embedding method that is used to embed diversity coefficients and the lower/upper bounds across tasks would aid in usage of these bounds across modalities.
		- but we hypothesize that CLIP/aligned embeddings are enough. Coming up with a L/U bound for vision can be useful to further validate/test/examine our currently proposed methods
- To really clarify why FIM is better than activations we provide this intuitive explanation. FIM gives a weight/feature of which parameter of the generative distribution matters (e.g. the first coordinate of Task2Vec corresponds to how artsy the text sequence is,). This is a feature of a task or dataset itself. Therefore, FIM exactly approximate the (task) data generative distribution we are trying to embed. Therefore, it results in superior representations for datasets compared to activations since it directly approximates the data (or task) generative distribution. 
```
My improved discussion section be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication.
Do not change citations e.g. \citep{...}, urls.
Do not copy the facts but the style of the top abstract example I provided.
Do not change the facts in my example.
Do not change any part that is already excellent.
Do not sound exaggerated or pompous.
Keep it concise, scientific, direct, use the active voice.
Follow the instructions conditioned by the example provided above.
The discussion section should be similar in length to the top example i.e. 1, 2 or 3 paragraphs.
Provide 3 re-phrased options:

# Prompt:



## Great outputs I choose:
TODO


# -- Notes --
Summary
```markdown
Discussion: Highlight key results and then proceeds to discusses implication (thus the work's potential impact), benefits, limitations, interpretations, compare with relevant literature, and future work (can be conected to impact and implications).
Conclusion: Provide a concise conclusion that summarizes your main findings, their implications, and potential future directions. No new information or arguments should be presented in the conclusion.
```

```markdown
What are the best practices or advice for writing a top quality machine learning discussion section for a research paper
on the caliber of NeurIPS, NIPS, ICML, ICLR, CVPR, AAAI? 

Write it out:
ChatGPT
The discussion section of a machine learning research paper is crucial to comprehensively convey the implications of your findings, provide interpretations, draw conclusions, and outline potential directions for future work. Here are the best practices to write a top-quality machine learning discussion section:

Highlight key results: Begin by succinctly summarizing the most significant results of your research. State these results in the context of the objectives of your study. Highlight the advancements over the existing state-of-the-art methods. Provide statistics and results from your experiments to back these assertions.

Interpretation of results: Don't just state your findings, but also interpret them. Discuss why the results turned out as they did and what they mean in the context of your research question. Consider using visualizations to help explain complicated outcomes.

Compare with relevant literature: Discuss how your findings align or contrast with previously published studies. This comparison not only shows the novelty and significance of your research but also embeds your work in the larger academic discourse.

Discuss limitations: A transparent and objective discussion should include the limitations of your work. Every algorithm or model has limitations, and it's important to acknowledge these. Discuss the potential weaknesses of your methodology, data, or theoretical framework, and how these limitations could have impacted the results.

Implications of your research: Discuss the implications of your findings on the field of machine learning. This might include implications for theoretical development, practical applications, or broader impacts, such as societal or ethical considerations.

Future work: Identify areas for future research, whether it's improving the current model, applying the model to other problems, or addressing the limitations of your current study. This helps in setting the direction for other researchers who might be interested in your work.

Conclusion: Provide a concise conclusion that summarizes your main findings, their implications, and potential future directions.

Clarity and readability: The discussion should be written in clear, precise, and professional language. Be logical in your argumentation and make sure that the text flows naturally.

Remember that the discussion section is not merely a restatement of the results, but a comprehensive, critical analysis of the research findings, providing meaning and context to these findings.
```
```markdown
so what is the difference between a conclusion and discussion section?
ChatGPT
In academic writing, the discussion and conclusion sections have distinct purposes and should contain different types of content.

Discussion: This is typically a detailed section where the results are interpreted and connected back to the original research question or hypothesis. 
It's where you can go in-depth into the implications of your results, compare them to other studies, discuss potential 
limitations, and propose ideas for future research. 
In essence, the discussion section is where you explain what your results mean in a broad context and in terms of your specific area of research.

Conclusion: The conclusion section is typically more concise and serves as a summary of your main findings and their implications. 
In contrast to the discussion, the conclusion should not introduce new topics or arguments. 
Instead, it should provide closure to your paper, summarizing the main points, reaffirming the importance of your work, 
and potentially indicating the wider implications of your findings. The conclusion is also where you can briefly mention
future directions for research, but without the detailed discussion and analysis that would be in the discussion section.

The distinction between these sections can sometimes blur, and in some papers, they may be combined into one "Discussion and Conclusion" section. 
However, even in this case, the roles of discussing and concluding should be clear in the text. 
It's also important to follow the specific guidelines of the journal or conference you are submitting to, as they may have particular preferences for how these sections should be structured.
```
```markdown
Brando — 
Hi professor, 
Curious on your perspective. I find that main results & conclusions are often combined in the discussion section. 
Sometimes I'm not sure what the point of a conclusion section is if the conclusion is stated in the discussion section. What is your perspective in general? (with the caveat that the answer likely depends on the paper)
sanmi — 
Agreed, I think it's fine to combine. One usually does not need both.
```
```markdown
6 Discussion (LITE, memory efficient meta-learning)
We propose LITE, a general and memory-efficient episodic training scheme for meta-learners that
enables them to exploit large images for higher performance with limited compute resources. LITE’s
significant memory savings come from performing a forward pass on a task’s full support set, but
back-propagating only a random subset, which we show is an unbiased estimate of the full gradient.
We demonstrate that meta-learners trained with LITE are state-of-the-art among meta-learners on
two challenging benchmarks, ORBIT and VTAB+MD, and are competitive with transfer learning
approaches at a fraction of the test time computational cost.
This offers a counterpoint to the recent narrative that transfer learning approaches are all you need for
few-shot classification. Both classes of approach are worthy pursuits (and will need to exploit large
images in real-world deployments) but careful consideration should be given to the data and compute
available at test time to determine which class is best suited to the application under consideration. If
it involves learning just a single task type (e.g. classifying natural images) with ample data and no
compute or time constraints, then a fine-tuning approach would suffice and perform well. However, if
a multitude of task types will be encountered at test time, each with minimal data, and new tasks need
to be learned on resource-constrained devices (e.g. a mobile phone or a robot) or quickly/repeatedly
(e.g. in continual or online learning settings), then a meta-learning solution will be better suited.
Finally, as the machine learning community grapples with greener solutions for training deep neural
networks, LITE offers a step in the right direction by allowing meta-learners to exploit large images
without an accompanying increase in compute. Future work may look toward applying the basic
concept of LITE to other types of training algorithms to realize similar memory savings.
Limitations As discussed in Section 3, LITE can be applied to a wide range of meta-learners
provided that they aggregate the contributions from a task’s support set via a permutation-invariant
operation like a sum. Because only a subset of the support set is back-propagated, however, the
gradients can be more noisy and meta-training may require lower learning rates. Furthermore, LITE
is a memory-efficient scheme for training meta-learners episodically and has not been tried with
meta-learners trained in other ways (e.g. with standard supervised learning) or non-image datasets.
Societal impact Few-shot learning systems hold much positive potential – from personalizing
object recognizers for people who are blind [14] to rendering personalized avatars [30] (see [23]
for a full review). These systems, however, also have the potential to be used in adverse ways – for
example, in few-shot recognition in military/surveillance applications. Meta-trained few-shot systems
may also pose risks in decision making applications as uncertainty calibration in meta-learning
models has not yet been extensively explored. Careful consideration of the intended application, and
further study of uncertainty quantification in meta-learning approaches will be essential in order to
minimize any negative societal consequences of LITE if deployed in real-world applications.
```

Example with Q/A in discussion, actually this is not in the discussion section, but it's a good example of 
FAQ (ref https://arxiv.org/pdf/1909.02729.pdf):
```latex
% itmenize so that question are numbered/easy to reference
\textbf{Why has it not been noticed yet that this simple approach works so well?}
Non-transductive fine-tuning as a baseline has been considered before (Vinyals et al., 2016; Chen
et al., 2018). The fact that this is comparable to state-of-the-art has probably gone unnoticed
because of the following reasons:
• Given that there are only a few labeled support samples provided in the few-shot setting,
initializing the classifier becomes important. The support-based initialization (cf. Section 3.1)
motivated from the deep metric learning literature (Hu et al., 2015; Movshovitz-Attias et al.,
2017; Qi et al., 2018; Gidaris & Komodakis, 2018) classifies support samples correctly (for
a support shot of 1, this may not be true for higher support shots). This initialization, as
opposed to initializing the weights of the classifier randomly, was critical to performance in
our experiments.
• In our experience, existing meta-training methods, both gradient-based ones and metricbased ones, are difficult to tune for larger architectures. We speculate that this is the reason a
large part of the existing literature focuses on smaller backbone architectures. The few-shot
learning literature has only recently started to move towards bigger backbone architectures
(Oreshkin et al., 2018; Rusu et al., 2018). From Table 3 we see that non-tranductive finetuning gets better with a deeper backbone architecture. A similar observation was made
by (Chen et al., 2018). The observation that we can use “simple” well-understood training
techniques from standard supervised learning that scale up to large backbone architectures
for few-shot classification is a key contribution of our paper.
Transductive methods have recently started to become popular in the few-shot learning literature
(Nichol et al., 2018; Liu et al., 2018a). Because of the scarcity of labeled support samples, it is
crucial to make use of the unlabeled query samples in the few-shot regime.
Our advocated baseline makes use of both a good initialization and transduction, relatively new in
the few-shot learning literature, which makes this simplistic approach go unrecognized till now
```

rfs: 
```latex
\section{Discussion}
We have proposed a simple baseline for few-shot image
classification in the meta-learning context. This approach
has been underappreciated in the literature thus far. We
show with numerous experiments that such a simple baseline
outperforms the current state-of-the-arts on four widelyused few-shot benchmarks. 
Combined with self-distillation,
the performance further improves by 2-3%. Even when
meta-training labels are unavailable, it may be possible to
leverage state of the art self-supervised learning approaches
to learn very good embeddings for meta-testing tasks.

1. \textbf{Q: What is the intuition of this paper?}
A: We hope this paper will shed new light on few-shot classification. We believe representations play an important
role. Shown by our empirical experiments, a linear model
can generalize well as long as a good representation of the
data is given.

2. Why does this simple baseline work? Is there anything
that makes few-shot classification special?
A: Few-shot classification is a special case of meta-learning
in terms of compositionality of tasks. Each task is an Kway classification problem, and on current benchmarks the
classes, even between tasks, are all mutually exclusive. This
means we can merge all N of the K-way classification tasks
into a single but harder NK-way classification task. Our
finding is that training an embedding model on this new
NK-way task turns out to transfer well to meta-testing set.
On the other hand, we also find that self-supervised embedding, which does not explicitly require this NK compositionality, achieves a similar level of performance. A concurrent work [10] studies the representations for few-shot
learning from the theoretical point of view.

3. Does your work negate recent progress in meta-learning?
A: No. Meta-learning is much broader than just fewshot classification. Although we show a simple baseline
outperforms other complicated meta-learning algorithms
in few-shot classification, methods like MAML may still
be favorable in other meta-learning domains (e.g., metareinforcement learning).

4. Why does distillation work? What does it suggest?
A: The soft-labels [18] from the teacher model depict the
fact that some classes are closer to each other than other
classes. For example, a white goat is much more similar
to a brown horse than to an airplane. But the one-hot label
does not capture this. After being regularized by soft-labels,
the network learns to capture the metric distance. From theoretical perspective, [36] provides analysis for linear case.
Ongoing work [31] argues distillation amplifies regularization in Hilbert space.
```