"""
ref: https://chatgpt.com/g/g-cH94JC5NP-dspy-guide-v2024-2-7
"""
#%%
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the local LLaMA model and tokenizer using Hugging Face's `transformers`.
# Replace 'meta-llama/Llama-2-7b-hf' with the local path or Hugging Face model you are using.
# model_name = "meta-llama/Llama-2-7b-hf"  # Adjust model name as necessary
model_name = "gpt2"  # Adjust model name as necessary

# Load the model and tokenizer
local_model = AutoModelForCausalLM.from_pretrained(model_name)
local_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Configure DSPy to use the local LLaMA model as the LM.
huggingface_lm = dspy.HuggingFaceLM(model=local_model, tokenizer=local_tokenizer)
dspy.settings.configure(lm=huggingface_lm, rm=None)  # No retrieval model used

# Step 3: Define a small, high-quality hardcoded dataset (3-5 examples).
# These examples are used for training few-shot learning with factual answers.
train_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},              # Simple factual QA pair
    {"question": "Who wrote '1984'?", "answer": "George Orwell"},                   # Author-related factual question
    {"question": "What is the boiling point of water?", "answer": "100°C"},         # Scientific fact question
]

# Dev set for evaluating model generalization on unseen examples.
dev_data = [
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},      # Historical factual question
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},               # Simple factual QA pair for testing
]

# Convert the dataset into DSPy examples with input/output fields.
trainset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in train_data]
devset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in dev_data]

# Step 4: Define the RAG program (Simple QA Generation).
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought generates answers using the configured local LM (LLaMA in this case).
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        # Pass the question through the local LLaMA LM to generate an answer.
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)

# Step 5: Metric to evaluate exact match between predicted and expected answer.
def exact_match_metric(example, pred, trace=None):
    return example['answer'].lower() == pred.answer.lower()

# Step 6: Use teleprompter (BootstrapFewShot) to optimize few-shot examples for the best performance.
# It optimizes the examples selected from the train set based on the exact match metric.
teleprompter = BootstrapFewShot(metric=exact_match_metric)

# Compile the SimpleQA program with optimized few-shots from the train set.
compiled_simple_qa = teleprompter.compile(SimpleQA(), trainset=trainset)

# Step 7: Test with a sample question and evaluate the performance.
my_question = "What is the capital of Japan?"
pred = compiled_simple_qa(my_question)

# Output the predicted answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

# Evaluate the compiled program on the dev set using the exact match metric.
evaluate_on_dev = Evaluate(devset=devset, num_threads=1, display_progress=False)
evaluation_score = evaluate_on_dev(compiled_simple_qa, metric=exact_match_metric)

print(f"Evaluation Score on Dev Set: {evaluation_score}")

#%%
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the local model and tokenizer using Hugging Face's `transformers`.
# Replace 'gpt2' with any local model that fits your resources.
model_name = "gpt2"  # Use a locally installed model

# Load the model and tokenizer from Hugging Face's transformers library.
local_model = AutoModelForCausalLM.from_pretrained(model_name)
local_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Configure DSPy to use the local Hugging Face model as the LM.
huggingface_lm = dspy.HuggingFaceLM(model=local_model, tokenizer=local_tokenizer)
dspy.settings.configure(lm=huggingface_lm, rm=None)  # No retrieval model used

# Step 3: Define a small, high-quality hardcoded dataset (3-5 examples).
# These examples are used for training few-shot learning with factual answers.
train_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},              # Simple factual QA pair
    {"question": "Who wrote '1984'?", "answer": "George Orwell"},                   # Author-related factual question
    {"question": "What is the boiling point of water?", "answer": "100°C"},         # Scientific fact question
]

# Dev set for evaluating model generalization on unseen examples.
dev_data = [
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},      # Historical factual question
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},               # Simple factual QA pair for testing
]

# Convert the dataset into DSPy examples with input/output fields.
trainset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in train_data]
devset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in dev_data]

# Step 4: Define the RAG program (Simple QA Generation).
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought generates answers using the configured local LM.
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        # Pass the question through the local LM to generate an answer.
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)

# Step 5: Metric to evaluate exact match between predicted and expected answer.
def exact_match_metric(example, pred, trace=None):
    return example['answer'].lower() == pred.answer.lower()

# Step 6: Use teleprompter (BootstrapFewShot) to optimize few-shot examples for the best performance.
# It optimizes the examples selected from the train set based on the exact match metric.
teleprompter = BootstrapFewShot(metric=exact_match_metric)

# Compile the SimpleQA program with optimized few-shots from the train set.
compiled_simple_qa = teleprompter.compile(SimpleQA(), trainset=trainset)

# Step 7: Test with a sample question and evaluate the performance.
my_question = "What is the capital of Japan?"
pred = compiled_simple_qa(my_question)

# Output the predicted answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

# Evaluate the compiled program on the dev set using the exact match metric.
evaluate_on_dev = Evaluate(devset=devset, num_threads=1, display_progress=False)
evaluation_score = evaluate_on_dev(compiled_simple_qa, metric=exact_match_metric)

print(f"Evaluation Score on Dev Set: {evaluation_score}")

#%%
"""
docker run --gpus all --shm-size 1g -p 8080:80 -v $PWD/data:/data -e HUGGING_FACE_HUB_TOKEN={your_token} \
ghcr.io/huggingface/text-generation-inference:latest --model-id meta-llama/Llama-2-7b-hf --num-shard 1
"""
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate

# Step 1: Configure DSPy to use the local LLaMA model running on a TGI server.
# The server is hosted locally at port 8080.
tgi_llama2 = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
dspy.settings.configure(lm=tgi_llama2)

# Step 2: Define a small, high-quality hardcoded dataset (3-5 examples).
train_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote '1984'?", "answer": "George Orwell"},
    {"question": "What is the boiling point of water?", "answer": "100°C"},
]

# Dev set for evaluating model generalization on unseen examples.
dev_data = [
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
]

# Convert the dataset into DSPy examples with input/output fields.
trainset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in train_data]
devset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in dev_data]

# Step 3: Define the Simple QA program using DSPy.
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought generates answers using the configured local LLaMA LM via TGI.
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        # Pass the question through the local LM (LLaMA) to generate an answer.
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)

# Step 4: Metric to evaluate exact match between predicted and expected answer.
def exact_match_metric(example, pred, trace=None):
    return example['answer'].lower() == pred.answer.lower()

# Step 5: Use the teleprompter (BootstrapFewShot) to optimize few-shot examples.
teleprompter = BootstrapFewShot(metric=exact_match_metric)

# Compile the SimpleQA program with optimized few-shots from the train set.
compiled_simple_qa = teleprompter.compile(SimpleQA(), trainset=trainset)

# Step 6: Test with a sample question and evaluate the performance.
my_question = "What is the capital of Japan?"
pred = compiled_simple_qa(my_question)

# Output the predicted answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

# Evaluate the compiled program on the dev set using the exact match metric.
evaluate_on_dev = Evaluate(devset=devset, num_threads=1, display_progress=False)
evaluation_score = evaluate_on_dev(compiled_simple_qa, metric=exact_match_metric)

print(f"Evaluation Score on Dev Set: {evaluation_score}")
