# ---- Write Experiment & Results section ----
```text
references: 
    - memory efficient meta-learning: https://proceedings.neurips.cc/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html
```

# Prompt: experiment description
Rewrite my experiment description into top scientific machine learning experiment section, 
in the style of the top experiment example I will provide.
My experiment description:
```text
My experiment notes:
Among the five sub-datasets of The Pile, NIH ExPorter, PubMed Abstracts, and USPTO have low diversity (0.15-0.17).
This may be explained by dataset composition, since these datasets were curated from documents of a specialized field. 
For instance, NIH ExPorter and PubMed Abstracts are medicine-focused. 
Therefore, these datasets may contain sequences written in specific formats and prose, e.g. patent backgrounds in USPTO may share similar formats and semantics as do abstracts in NIH ExPorter or PubMed Abstracts. 
Therefore, sequences from these datasets may be more similar structurally and semantically.

Pile-CC and HackerNews, both of which cover a broader variety of topics, have higher diversity. Between the two, Pile-CC has higher diversity, which is consistent with its composition. We expect a greater variety of text topics, document formats, etc. from a general web scrape such as Pile-CC vs. a computer science and entrepreneurship-focused website such as HackerNews.

These results show that combining tasks from two different datasets computes a higher diversity coefficient. This aligns with our expectation that combining data from different sources increases the overall diversity of the data.
```
Top experiment example only use their style:
```text
Top experiment example description:
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

General outline:
\textbf{Experiments:} *describe* the actual experiments objectively (especially in 1st sentence).
\textbf{Results:} *describe* the results objectively and evaluation metrics 
(reference of supporting evidence e.g. tables, figures, etc. likely goes here).
Then explain *the key observations with out interpretations*, 
with comments relating to conclusions, interpretations, implications, and contributions. 
```
Your experiment section should be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication
(do not change citations e.g. \citep{...}, urls or names).
Do not copy the facts but the style of the top experiment section example I provided.
Also, do not change any part that is already excellent.
Do not sound exaggerated or pompous.
Keep it concise, scientific, direct, use the active voice.
Provide 3 re-phrased options:

## Great outputs I choose:
TODO
