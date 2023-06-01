# ---- Write Experiment & Results section ----
```text
references: 
    - memory efficient meta-learning: https://proceedings.neurips.cc/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html
```

# Prompt: experiment and results description
Rewrite my experiment and results description into top scientific machine learning experiment & results section, 
in the style of the top experiment & results example I will provide.
My experiment & results description or notes:
```text
\textbf{Experiments} 
We compute the diversity coefficient (section \ref{methods}) of eight publicly available LLM pre-training datasets (section \ref{section:datasets}).
We also compute the diversity coefficient of the five dataset concatenations (section \ref{methods_div_coeff_for_dataset_contact}).
In addition, we compute our conceptually well-motivated lower and upper bounds on the diversity coefficient (section \ref{method_ub_lb}).
% Their concatenations in section \ref{section:datasets}.

\textbf{Results}
Table \ref{table:div} reports the measured diversity coefficients of seven publicly available LLM pre-training datasets, 
in addition to the well-motivated lower and upper bounds.
Table \ref{table:div} also reports the measured diversity coefficient of the concatenation of different publicly available datasets. 
The key observations from the results are:
% make the part bellow a bullet point (itemize, latex) list as part of the key observations
- LLM datasets have diversity coefficients: \textbf{A. 3-5 times that of the conceptual lower bound and 
B. on average, half that of the upper bound.} In particular, WikiText-103, c4, The Pile, and Pile-CC have high diversity coefficients $(0.21, 0.25)$.
- Interestingly, Pile-CC has higher measured diversity than c4, suggesting that the preprocessing applied to Common Crawl corpus for Pile-CC may have been more rigorous in producing higher more diversequality data. 
% (for instance, in removing duplicate or near-duplicate sequences).
- Among the five sub-datasets of The Pile, NIH ExPorter, PubMed Abstracts, and USPTO have low diversity (0.15-0.17) 
i.e. half of the upper bound 0.4.
This may be explained by dataset composition, since these datasets were curated from documents of a specialized field. 
For instance, NIH ExPorter and PubMed Abstracts are medicine-focused. 
Therefore, these datasets may contain sequences written in specific formats and prose, 
e.g. patent backgrounds in USPTO may share similar formats and semantics as do abstracts in NIH ExPorter or PubMed Abstracts.
Therefore, text sequences from these datasets may be more similar structurally and semantically due to their technicality, resulting in low diversity.
- However, we observe that the Pile-CC and HackerNews have higher diversity.
We attribute it to the fact that both cover a broader variety of topics. 
Between the two, Pile-CC has higher diversity, which is consistent with its heterogenous composition. 
We expect a greater variety of text topics, document formats, etc. from a general web scrape such as Pile-CC vs. a computer science and entrepreneurship focused website.
```

Great examples of top experiment & results, only use their style (not their facts):
```text
<eg>
Experiments We meta-train ProtoNets [3], CNAPs [4] and Simple CNAPs [5] with LITE on tasks
composed of large (224 × 224) images. We also meta-train first-order MAML on large images as a
baseline. Since first-order MAML can process task support sets in batches, we simply reduce the
batch size and do not need to use LITE. We compare all of the above to meta-training on tasks of
small (84 × 84) images (i.e. the original baselines [14]). We also include a transfer learning approach,
FineTuner [28], which freezes a pre-trained feature extractor and fine-tunes just the linear classifier
for 50 optimization steps. For each model, we consider a ResNet-18 (RN-18) and EfficientNet-B0
(EN-B0) feature extractor, both pre-trained on ImageNet [29]. We follow the task sampling protocols
described in [14] (see Appendices B and C.1 for details). We also include analyses on meta-training
with small tasks of large images in Appendix D.3.

Results In Table 1, we report frame accuracy and video accuracy, averaged over all the query videos
from all tasks across all test users (17 test users, 85 tasks in total), along with their corresponding
95% confidence intervals. We also report the computational cost to learn a new task at test time in
terms of the number of Multiply-Accumulate operations (MACs), the number of steps to adapt, and
the wall clock time to adapt in seconds. See Appendix C.1 for results on additional metrics. 
The key observations from our results are:
• Training on larger (224 × 224) images leads to better performance compared to smaller (84 × 84)
images. The boost is significant for both clean and clutter videos, though absolute performance
remains lower on clutter videos. This suggests that object detection or other attention-based
mechanisms may be required to further exploit large images in more complex/cluttered scenes.
• All meta-learners + LITE set a new state-of-the-art on clean videos, and perform competitively
with the FineTuner on cluttered videos, using an EfficientNet-B0 backbone.
• Meta-learners are competitive with transfer learning approaches in accuracy but are almost two
orders of magnitude more efficient in the number of MACs and the time to learn a new task, and
one order of magnitude smaller in the number of steps to adapt.
</eg>
```

Also use the general outline below to rewrite my notes:
```text
\textbf{Experiments:} *describe* the actual experiments objectively (especially the 1st sentence).
Be specific.
\textbf{Results:} *describe* the results objectively and evaluation metrics 
(reference of supporting evidence e.g. tables, figures, etc. likely goes here).
Then explain *the key observations with out interpretations*, 
with comments relating to conclusions, interpretations, implications, and contributions.
If possible with key observations, as a bullet point list (as part of the results).
```

Your experiment section should be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication
(do not change citations e.g. \citep{...}, urls or names).
Do not copy the facts but the style of the top experiment section example I provided.
Also, do not change any part that is already excellent.
Do not sound exaggerated or pompous.
Keep it concise, scientific, direct, use the active voice.
Provide 3 options:

## Great outputs I choose:
Great output 1:
```text
\textbf{Experiments} 
textbf{Experiments} 
We compute the diversity coefficient (section \ref{methods}) of eight publicly available LLM pre-training datasets (section \ref{section:datasets}).
We also compute the diversity coefficient of the five dataset concatenations (section \ref{methods_div_coeff_for_dataset_contact}).
In addition, we compute our conceptually well-motivated lower and upper bounds on the diversity coefficient (section \ref{method_ub_lb}).

\textbf{Results}
Table \ref{table:div} reports the measured diversity coefficients of seven publicly available LLM pre-training datasets, 
in addition to the well-motivated lower and upper bounds.
Table \ref{table:div} also reports the measured diversity coefficient of the concatenation of different publicly available datasets. 
The key observations from the results are:
\begin{itemize}[topsep=-5pt, itemsep=-5pt, leftmargin=*]
    \item The diversity coefficient of pre-training datasets tend to be \textbf{3-5 times greater than the theoretical lower bound and, on average, half the upper bound.} 
    Prominently, WikiText-103, c4, The Pile, and Pile-CC exhibit high diversity coefficients (0.21, 0.25).
    \item The measured diversity of Pile-CC is higher than that of c4, indicating a potentially more stringent preprocessing method applied to the Common Crawl corpus for Pile-CC, which contributes to enhanced data diversity.
    \item Three sub-datasets of The Pile, namely NIH ExPorter, PubMed Abstracts, and USPTO, show relatively low diversity (0.15-0.17), approximately half of the upper bound 0.4. The nature of these datasets, curated from specialized fields, may account for this observation.
    \item However, we observe that the Pile-CC and HackerNews have higher diversity, which may be attributed to their coverage of a broad range of topics. 
    Among these, Pile-CC exhibits higher diversity, in line with its heterogeneous content composition.
\end{itemize}
```


