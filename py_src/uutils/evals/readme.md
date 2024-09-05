# Running Check Final Answer Evaluations for Language Models (LMs)

## Sanity Checking Our Eval code with Mistral7B
This code is a small adaptation from the [Meta-Math](https://meta-math.github.io/) original evaluation. 
We have verified that it runs within `1-2%` (Redo, currently code doesn't run with n=4 so can't reproduce for now) accuracy difference with Mistral7B-base, therefore giving us confidence this code is correct and reliable to use. 
<!-- Note mistral ins 13.1% ref: https://mistral.ai/news/announcing-mistral-7b/ us on MATH TODO, lost value sadly -->
Mistral Instruct got **13.1%** on MATH, as reported [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), 
and we got **X** on MATH. 
Run the code bellow to reproduce/sanity check.  
The following run has HPs that worked in an A100 40GB machine: 
```bash
# - Ins
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=3
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-Instruct-v0.1 --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --batch_size 5000 --mode dryrun
# - Base
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=4
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-v0.1 --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --batch_size 5000 --mode dryrun
```
<!-- TODO: edit prompt to look like mistral's (or meta maths), likely closes gap btw two results. 
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
Output:
```bash
TODO
``` -->

| Model             | MATH Benchmark (Our Eval)                        | MATH Benchmark (Their Eval)                      | Runtime (Our Eval) (Num Examples) |
|-------------------|--------------------------------------------------|--------------------------------------------------|-----------------------------------|
| Mistral Base      | 3.00% (0-shot CoT, maj@1)                        | n/a                                              | 64.39 mins (1.07 hr) (5000)       |
| Mistral Ins       | 5.36% (0-shot CoT, maj@1, wrong ins)             | 13.1% (4-shot CoT, maj@4)                        | 42.66 mins (0.71 hr) (5000)       |
|-------------------|--------------------------------------------------|--------------------------------------------------|-----------------------------------|
| Mistral Ins       | 6.62% (8-shot CoT, maj@1, right ins)             | 13.1% (4-shot CoT, maj@4)                        | 22.66 mins (0.38 hr) (5000)       |
| Mistral Ins       | 7.4% (8-shot CoT, maj@1, right ins)              | 13.1% (4-shot CoT, maj@4)                        | 2.69 mins (0.04 hr)  (500)        |
|-------------------|--------------------------------------------------|--------------------------------------------------|-----------------------------------|
<!-- | DSC 7B Base       | 0.04% (0-shot CoT, maj@1)                        | n/a                                              | 128.46 mins (2.14 hr) (5000)      | -->
<!-- |-------------------|--------------------------------------------------|--------------------------------------------------|-----------------------------------| -->
<!-- | Claude 3.5 Sonnet | 6.62% (0-shot CoT, maj@1, Mistral Ins Prompt)    | n/a                                              | 22.5 mins (0.37 hr) (5000)        | -->

*Note:* 
- **maj@1**: Majority voting across 1 sample (single prediction).
- **maj@4**: Majority voting across 4 samples (four predictions per question, and the most common answer is chosen as the final prediction).
- **wrong ins**: Indicates that the official formatting for the prompt was not used at inference, which needs to be fixed (TODO).
- **right ins**: Indicates that the correct instruction format was used.
- **DSC 7B Base v1.5**: [Deepseek-Coder-7B-Base-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-base-v1.5).


TODO: perhaps some day if relevant why Claude and DSC had such bad performance with out code. 

<!-- https://chatgpt.com/c/934e88a3-e8df-45cc-8b47-427dd651150f for table -->


```bash
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=4
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-Instruct-v0.1 --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 1024 --batch_size 500 --end 500 --n 1 --shuffle True --mode dryrun

source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=5
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-Instruct-v0.1 --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 1024 --batch_size 5000 --end 5000 --n 1 --shuffle True --mode dryrun
```

## Sanity Checking Our Eval code with Claude 3.5 Sonnet
We also did a check with Claude 3.5 Sonnet and [the original Anthropic blog](https://www.anthropic.com/news/claude-3-5-sonnet) reports `71.1%` with `0-shot CoT` on Hendryck's MATH eval benchmark. 
Claude 3 Opus reports `60.1%` with `0-shot Cot` on Hendryck's MATH eval benchmark.. 
Using our own `0-shot Cot` (note: it's impossible to know exactly their prompt and setting) we got `X` result using our own eval code on Hendryck's MATH eval benchmark. 
To verify Claude accuracy run:
```bash
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model claude-3-5-sonnet-20240620 --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --mode dryrun
```
Output:
```bash
TODO
```

# Install

Create a venv or conda env for this project, instructions here or use these simplified set of instructions:
```bash
# - Create conda env (note: vllm has issues with 3.10 so we are using 3.9, ref: https://gist.github.com/brando90/c55c74e840d42c952d4aec7b74e0be6c)
# conda create -n snap_cluster_setup_py3_9 python=3.9
conda create -n snap_cluster_setup python=3.11
# - Activate your conda env
conda activate snap_cluster_setup
# - Pip install snap-cluster-setup repo in editable mode with pip
pip install --upgrade pip
pip install -e ~/snap-cluster-setup
```
If using vLLM (see issues of installations [here](https://github.com/vllm-project/vllm/issues/2747)):
```bash
# - If using GPUs (non Propriety models)
pip install --upgrade pip
pip install torch==2.2.1
pip install vllm==0.4.1
```

Verify the data has the right number of points (200 August 14 2024):
```bash
jq -c '.[]' /~/putnam-math/data/Putnam_MATH_original_static_final/Putnam_MATH_boxed_problems.json | wc -l
```
Sample output:
```bash
200
```

# Quickstart: Open Source Model Putnam Evaluations
The instructions here are for reproducing our Open Source model evaluations based on [this early version of the manuscript](https://openreview.net/forum?id=1720vDqiBK#discussion) 
and [the results form this table](py_src/evals/eval_images/first_results_putnam_math.png).

Select a GPU:
```bash
export CUDA_VISIBLE_DEVICES=1 
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
```

Now give the model and path to the data set you want to evaluate [in the format of Hendryck's MATH data set](https://github.com/hendrycks/math):
```bash
# - Mistral 7B Base (https://huggingface.co/mistralai/Mistral-7B-v0.1)
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-v0.1 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test
# --> Uncomment for newer versions of the data set
# python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model mistralai/Mistral-7B-v0.1 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static_final

# - LLama 3 8B Base (https://huggingface.co/meta-llama/Meta-Llama-3-8B)
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test

# - Gemma 2B
python boxed_acc_eval.py --model google/gemma-2b --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test 

# - Deep-Seek-Math 7B Base (https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-base --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 1 --mode online 

# - Deep-Seek-Math 7B Instruct (https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-instruct --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 1 --mode online 

# - Deep-Seek-Math 7B RL (https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-rl --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 1 --mode online 

```

## Quickstart - Proprietry Model Putnam Evaluations
The instructions here are for reproducing our Prioprietry Source model evaluations based on [this version of the early manuscript](https://openreview.net/forum?id=1720vDqiBK#discussion) 
and [the results form this table](py_src/evals/eval_images/first_results_putnam_math.png).

### OpenAI GPT Evaluations
The following are the commands to run GPT evaluations. 
Tip: use GPT3.5 (or the chepaer version when you read this) to **quickly** and **cheaply** verify everything is working for you before running the larger evaluations (~200 data points as of this writing):
```bash
# - GPT 3.5
python boxed_acc_eval.py --model gpt-3.5-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 
# python boxed_acc_eval.py --model gpt-4o-mini --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 

# - GPT 4 Turbo
python boxed_acc_eval.py --model gpt-4-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348

# - GPT 4o
python boxed_acc_eval.py --model gpt-4o --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 
```

### Anthropic Claude Evaluations
The following are the commands to run [Anthropic's Claude](https://docs.anthropic.com/en/docs/about-claude/models) evaluations. 
```bash
# - Claude 3 Opus

# - Claude 3.5 Sonnet 
# python boxed_acc_eval.py --model  --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static2/test --end 348 --batch_size 348 --mode dryrun
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model claude-3-5-sonnet-20240620 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static_final --end 348 --batch_size 348 --mode dryrun
# python boxed_acc_eval.py --model claude-3-5-sonnet-20240620 --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_original_static_final/Putnam_MATH_boxed_problems.json --end 348 --batch_size 348 --mode dryrun
```
**Result Claude 3.5 Sonnet:** get's `16/200=8%` right 0-shot CoT mv@4 on `Putnam_MATH_original_static_final` 08/15/2024. 

### Gemini Evaluations
TODO:

# Evaluations with the Variations Benchmarks

### Generation of Benchmark Datasets with our Python Scripts 
TODO

### Open Source Model Evluations on the Variation Benchmarks
```bash
# python boxed_acc_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model deepseek-ai/deepseek-math-7b-base --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
```

### Proprietrg Model Evluations on the Variation Benchmarks
```bash
# python boxed_acc_eval.py --model gpt-3.5-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model gpt-4-turbo --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
# python boxed_acc_eval.py --model gpt-4o --path_2_eval_dataset ~/putnam-math/data/Putnam_MATH_variations_static2/test --end 348 --batch_size 348 --mode online 
```

# Running with Many backends -- HF Pipeline and vLLM
## Running Eval with vLLM - 1.8B model example
Running on InternLM-2.5-1.8B. 
The guessed max length is based on [this paragraph](https://huggingface.co/internlm/internlm2_5-1_8b/discussions/2) for public PT checkpoint:
> ChatGPT (RAG): InternLM2 was initially trained on a context length of **4096** token ... extended contexts efficiently, with the ability to process up to 32k tokens during training 
The following run has HPs that worked in an A100 40GB machine:
```bash
# - Evaluate InternLM-2.5-1.8B from HF ckpt using vLLM backend
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=1
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model internlm/internlm2_5-1_8b --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --batch_size 5000 --mode dryrun
```
Output:
```bash
Processed prompts: 100%|███████████████| 5000/5000 [22:19<00:00,  3.73it/s]
...
wandb:          boxed_acc 0.1604
wandb:       len(results) 5000
wandb: len(results_boxed) 5000
wandb: sum(results_boxed) 802
```
At the time of this writing out eval code got **16.0%** while the original tech report for InternLM-2.5-1.8B reports **0.049**. 
Code with vLLM ran in 22 minutes. 

For SFT/CPT (Supervised Fine-Tuning, Continual Pre-trained) checkpoint we recommend you use the max length you used during that training. 
Replace the path with your path to the ckpt of course:
The following run has HPs that worked in an A100 40GB machine: 
```bash
# - Evaluate InternLM-2.5-1.8B from HF ckpt path using vLLM backend
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=2

# create copy of intermediate ckptn since HF deletes very old ones
cp -r ~/runs/run_08092024_internlm/internlm2-1_8b/train/checkpoint-60061 ~/runs/run_08092024_internlm/internlm2-1_8b/train/checkpoint-60061_copy
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model ~/runs/run_08092024_internlm/internlm2-1_8b/train/checkpoint-60061_copy --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --mode dryrun
```

## Running Eval with vLLM - 6.7B (~7B) model example
Running on [DeepSeek-Coder-V2-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base). 
This was the base model for DeepSeek-MATH-X family, so it's a good model to know the accuracy if you are trying to beat that model family. 
DeepSeek-Coder-V2-Base has an accuracy of **X** on the MATH data set. 
The following run has HPs that worked in an A100 40GB machine: 
```bash
# - Evaluate DeepSeek-Coder-V2-Base
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=1
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model deepseek-ai/deepseek-coder-7b-base-v1.5 --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --batch_size 5000 --mode dryrun
```
SFT/CPT example: 
```bash
# - Evaluate DeepSeek-Coder-V2-Base
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=2
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model ~/runs/run_08082024_deepseek-ai/deepseek-coder-7b-base-v1.5/train/checkpoint-40700_copy --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --batch_size 5000 --mode dryrun
```

## Running Eval with HF Pipeline
```bash
# - Evaluate InternLM-2.5-1.8B from HF
export CUDA_VISIBLE_DEVICES=1
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model internlm/internlm2_5-1_8b --hf_gen_type pipeline --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_length 4096 --batch_size 5_000 --mode dryrun
```
Give it your path
```bash
# - Evaluate InternLM-2.5-1.8B from HF ckpt path
export CUDA_VISIBLE_DEVICES=2
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model ~/runs/run_08082024/train/checkpoint-52500 --hf_gen_type pipeline --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_length 4096 -batch_size 5_000 --mode dryrun
```

### Saving Mode Responses
TODO

Motivation: debugging, human evaluations, and automatic proof evaluations e.g., Teacher Forced Accuracy (tfa).


### Development Check - Run Eval on 5 MATH examples
To check if the code is working (e.g., for devopment) run eval with 5 examples with [the small GPT2 (124M) model](https://huggingface.co/openai-community/gpt2) on GPU. 
The following run has HPs that worked in an A100 40GB machine but params are pretty small so it should work on most reasonable AI research hardware:  
```bash
# conda activate snap_cluster_setup
source ~/.virtualenvs/snap_cluster_setup/bin/activate
export CUDA_VISIBLE_DEVICES=3
python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model openai-community/gpt2 --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 4096 --batch_size 5 --end 5 --mode dryrun
```

### Flash Attn vLLM

```bash
# my current vllm setup without flash
# pip install --upgrade pip
# pip install torch==2.2.1
# pip install vllm==0.4.1

# flash attn https://amzn-aws.slack.com/archives/C06Q26TNN8G/p1724182667464149
# flash-attn>=2.5.8
# pip install flash-attn
# known setup to work with flash
# vllm                              0.5.4
# vllm-flash-attn                   2.6.1
# flash-attn                        2.6.3
# torch                             2.4.0
# Python 3.10.8 

# try to install flash attn in a new py env
python3.11 -m venv ~/.virtualenvs/flash_attn_test
source ~/.virtualenvs/flash_attn_test/bin/activate
pip install --upgrade pip
pip install -e ~/snap-cluster-setup

pip list | grep vllm
pip list | grep torch
pip list | grep flash-attn
pip list | grep vllm-flash-attn

# # didn't work
# pip install torch==2.2.1
# pip install vllm==0.4.1
# MAX_JOBS=4 pip install flash-attn --no-build-isolation --force

# this installed flash but vllm didn't say in it's output it was using it
pip install torch==2.4.0
pip install vllm==0.5.4
pip install flash-attn==2.6.3
pip install vllm-flash-attn==2.6.1

python ~/snap-cluster-setup/py_src/evals/boxed_acc_eval.py --model internlm/internlm2_5-1_8b --hf_gen_type vllm --path_2_eval_dataset ~/snap-cluster-setup/data/MATH/test --max_tokens 2048 --batch_size 100 --end 100 -n 1 --shuffle True --mode dryrun 2>&1 | tee $LOG_FILE && echo "Log file created at: $LOG_FILE"
```

### Verifying OlympiadBenchm Data

```bash
# Navigate to the directory containing the JSON files
cd /Users/miranebr/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024
# Initialize a total counter
total_count=0
# Count the number of dictionaries in each JSON file in the directory and add to the total
for file in *.json; do
    count=$(jq '. | length' "$file")
    echo "File: $file - Count: $count"
    total_count=$((total_count + count))
done
# Output the total count
echo "Total count of dictionaries across all JSON files: $total_count"
```
Ouput
```bash
File: OE_MM_maths_en_COMP_modified.json - Count: 150
File: OE_TO_maths_en_COMP_modified.json - Count: 675
Total count of dictionaries across all JSON files: 825
```