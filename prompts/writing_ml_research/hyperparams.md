# 

# Prompt: Help me make an excellent paragraph about hyperparameters (hps hp) about my experiments
Write a top quality hyper parameter (hp) paragraph for a NeurIPS NIPS ICML ICLR AAAI CVPR machine learning publication from my notes.
This is an excellent example:
```text
<excellent_example_of_hp_paragraph>
\textbf{Experimental Details for Resnet12 for MiniImagenet:}
We used the Resnet12 provided by \citep{rfs}.
We used the Adam optimizer with learning rate 1e-3 for both MAML and USL.
We used the same cosine scheduler as in \citep{rfs} for USL and no cosine scheduler for MAML. 
We trained the USL model for 186 epochs. 
We trained the MAML model for 37,800 episodic iterations (outer loop iterations).
We used a batch size of 512 for USL and a (meta) batch size of 4 for MAML.
For MAML we used an inner learning rate of 1e-1 and 4 inner learning steps.
We did not use first order MAML.
It took 1 day 17 hours 2 minutes 41 seconds to train USL to convergence with a single dgx A100-SXM4-40GB GPU.
The MAML model was trained with Torchmeta \citep{torchmeta} which didn't support multi gpu training when we ran this experiment, so we estimate it took 1-2 weeks to train on a single GPU.
In addition, it was ran with an earlier version of our code, so we unfortunately did not record the type of GPU but suspect it was either an A100, A40 or Quadro RTX 6000.
</excellent_example_of_hp_paragraph>

<excellent_example_of_hp_paragraph>
\textbf{Experimental hyperparameter details for Resnet12 on on low diversity data sets for Pre-training and fo/ho-MAML:}
We used the Resnet12 architecture provided by \citep{rfs}. 
The Adam optimizer \citep{adam} was utilized with a constant learning rate of 1e-3. 
No learning rate scheduler was used. 
Training was performed for 600,000 iterations for pre-training 
and 160,000 first-order MAML iterations, with a batch size of 256. 
The outer loop consisted of 130,000MAML iterations. 
Inner learning rate 0.1 and 5 inner steps.
No weight decay was applied. 
Training was performed on a single NVIDIA PU with at most 48GB memory select by a HPC automatically.
All experiments were trained to convergence (less than 0.01 loss) and took on average at most 1 week. 
All implementations were done in PyTorch \citep{pytorch}.
</excellent_example_of_hp_paragraph>

<excellent_example_of_hp_paragraph>
\textbf{Experimental Details for 5CNN on High Diversity Datasets:}
We utilized the 5CNN architecture proposed in \cite{rfs} with varying filter sizes. 
The Adam optimizer \cite{adam} was used with a learning rate of 1e-3 without any learning rate decay. 
A batch size of 256 was used for both pre-training and MAML training. 
No weight decay was applied. 
For pre-training, we trained for 200,000 iterations. 
For first-order MAML, we trained for 100,000 iterations with an inner loop of 5 steps and an inner learning rate of 0.1. 
We annealed the learning rate with a cosine scheduler with scheduler freq 2000 with minimum learning rate 1e-5 (similar to MAML++).
All models were trained to convergence, which took approximately 1 week on a single NVIDIA GPU with at least 48GB of memory allocated by the HPC scheduler. 
All implementations were done in PyTorch \cite{pytorch}.
</excellent_example_of_hp_paragraph>

<excellent_example_of_hp_paragraph>
\textbf{Hyperparameter Details for ResNet12 on High Diversity Benchmarks:}
We utilized the ResNet12 architecture from \citet{rfs} for our experiments.
The Adam optimizer \citep{adam} was used with a learning rate of 1e-3 without any learning rate decay.
For pre-training, we trained for 1 million iterations with a batch size of 256.
For first-order MAML \citep{maml}, we trained for 300,000 iterations also with a batch size of 256.
The MAML outer loop consisted of 5 inner update steps with an inner learning rate of 0.1.
No weight decay was applied.
We annealed the learning rate with a cosine scheduler with scheduler freq 2000 with minimum learning rate 1e-5 (similar to MAML++).
All models were trained to convergence on a single NVIDIA GPU with at least 48GB of memory allocated by the cluster scheduler.
Training took approximately 1-2 week to converge for both pre-training and MAML.
All implementations were done in PyTorch \citep{pytorch}.
</excellent_example_of_hp_paragraph>

<excellent_example_of_hp_paragraph>
\textbf{Hyperparameter Details for ResNet50 on High Diversity Meta-Dataset Benchmarks:}
We utilized the ResNet50 architecture from \citet{rfs} in our experiments on the high diversity meta-dataset benchmarks.
The Adafactor optimizer \citep{adafactor} was used with default settings and no learning rate decay.
For pre-training, we trained for 300,000 iterations with a batch size of 256.
For first-order MAML \citep{maml}, we trained for 140,000 iterations also with a batch size of 256.
The MAML outer loop consisted of 5 inner update steps with Adafactor's default inner learning rate.
We used Adafactor default annealing scheduler in seqfair.
Due to computational constraints, we limited the number of random seeds to 5 -- especially given that MDS combined 10 large scale vision data sets that includes ImageNet.
Pre-training and MAML training took approximately 1 month each to converge on NVIDIA GPUs with 48GB memory allocated automatically by the cluster scheduler.
All implementations were done in PyTorch \citep{pytorch}.
</excellent_example_of_hp_paragraph>
```
These are my notes:
```
<example_to_convert_to_excellent_hp_paragraph>
meta-data set (MDS) high div
adafactor
resnet50 as in RFS, reused their code
batch_size 256
pt iterations ~ at least 300K
maml iterations ~ at least 140K 
no weight decay
adafactor scheduler
these took ~1 month to run, hence the limitation of seeds
we used adafactor from seqfair because the it required no hyperparameter choices and there is previous work that showed the training was stable.
In addition adafactor uses less memory and this needed due to our GPU memory availability since 48GHs were not always available or easy to get for 1 month.
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
