# 

# Prompt: Help me make an excellent paragraph about hyperparameters (hps hp) about my experiments
Write a top quality hyper parameter (hp) paragraph for a NeurIPS NIPS ICML ICLR AAAI CVPR machine learning publication from my notes.
This is an excellent example:
```text
<excellent_example_of_hp_paragraph>
Experimental Details for Resnet12 for MiniImagent: We used the Resnet12 provided by [43]. We
used the Adam optimizer with learning rate 1e-3 for both MAML and USL. We used the same cosine
scheduler as in [43] for USL and no cosine scheduler for MAML. We trained the USL model for 186
epochs. We trained the MAML model for 37,800 episodic iterations (outer loop iterations). We used
a batch size of 512 for USL and a (meta) batch size of 4 for MAML. For MAML we used an inner
learning rate of 1e-1 and 4 inner learning steps. We did not use first order MAML. It took 1 day 17
hours 2 minutes 41 seconds to train USL to convergence with a single dgx A100-SXM4-40GB GPU.
The MAML model was trained with Torchmeta [12] which didn’t support multi gpu training when
we ran this experiment, so we estimate it took 1-2 weeks to train on a single GPU. In addition, it
was ran with an earlier version of our code, so we unfortun
</excellent_example_of_hp_paragraph>
```
These are my notes:
```
<example_to_convert_to_excellent_hp_paragraph>
adam optimizer
lr = 1e-3
batch_size 256
resnet12 as in rfs
no scheduler
no weight decay
pt iterations ~ at least 600k
fo maml iterations ~ at least 160k
ho maml iterations ~ at least 130K
</example_to_convert_to_excellent_hp_paragraph>
```
My improved discussion section be of top quality for a NeurIPS NIPs ICML ICLR machine learning publication. 
Do not change citations e.g. \citep{...}, urls. 
Do not copy the facts but the style of the top abstract example I provided. 
Do not change the facts in my example. 
Do not change any part that is already excellent. 
Do not sound exaggerated or pompous. 
Keep it concise, scientific, direct, use the active voice. 
Provide me the excellent paragraph:



----
Follow the instructions conditioned by the example provided above. 
