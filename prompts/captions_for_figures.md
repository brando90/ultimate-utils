# ---- Write Captions for Figures in a Research Paper ----
```text
references: 
    - (ANIL) Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML: https://arxiv.org/abs/1909.09157
```

# Prompt: create exccellent captions
Rewrite my caption notes into a top scientific machine learning caption:
```text
My caption notes:

```
```text
Top caption example:
Table 1: \textbf{Freezing successive layers (preventing inner loop adaptation) does not affect accuracy, supporting 
feature reuse.} 
To test the amount of feature reuse happening in the inner loop adaptation, we test the
accuracy of the model when we freeze (prevent inner loop adaptation) a contiguous block of layers at test time.
We find that freezing even all four convolutional layers of the network (all layers except the network head) hardly
affects accuracy. 
This strongly supports the feature reuse hypothesis: layers don’t have to change rapidly at
adaptation time; they already contain good features from the meta-initialization.

General outline of a great scientific caption:
\textbf{Bold the Main Contribution or Point or Conclusion of the figures.} 
Then some none bold text describing some details.
```
Your caption for the figure should be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication
(do not change citations e.g. \citep{...}, urls or names).
Do not copy the facts but the style of the top caption example I provided.
Also, do not change any part that is already excellent.
Do not sound exaggerated or pompous.
Keep it concise, scientific, direct, use the active voice.
Provide 20 re-phrased options:

## Great outputs I choose:
TODO
