# ---- Guidelines for Writing an Excellent Methods/Approach Section ----


## Prompt: 
Write a top quality machine learning research paper methods section using the current paragraphs and bullet points: 
```latex
% current notes of methods section
% goal: train given fixed compute budget a base model on DQ until we beat one of the 3 baselines
%    - (B0 bmdl [sanity], B1 hf leaderboard, B2 bmdl | handcrafed ds, B3 bmdl | DIRS).
% 1. fft on Falcon 7B
% - same hps (mdl arch, mdl size, batch size, opt, epoch(s)=1, thus compute)
% - generate a new data set from the whole (or refined) Flacon data set using batches (instead of per point)
% - train for a while so that you beat the best of same size in HF leader board
% - note: if we fft we can also test if increasing DQ improves the performance
```
Example of a good methods section:
```latex

```
Your methods section should be of top quality for a NeurIPS NIPs ICML ICLR AAAI CVPR machine learning publication
(do not change citations e.g. \citep{...} \cite{...}, urls or names).
Also, do not change any part that is already excellent.
Do not sound exaggerated or pompous.
Keep it concise, scientific, use the active voice.
Do not invent anything, just re-write the method I suggested into something that is described clearly.
Provide 5 re-phrased options: