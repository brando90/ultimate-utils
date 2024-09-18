"""
TODO: fix server running.
ask tom: https://github.com/stanfordnlp/dspy/issues/1002
ref: gpt and me https://chatgpt.com/c/66e4d31b-20e0-8001-a10f-7835d3b17182

ref: https://chatgpt.com/g/g-cH94JC5NP-dspy-guide-v2024-2-7

python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-hf --port 8080
"""
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate

# Step 1: Configure DSPy to use the local LLaMA model running on a vLLM server.
# The server is hosted locally at port 8080.
# vllm_llama2 = dspy.HFClientVLLM(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
# dspy.settings.configure(lm=vllm_llama2)
# dspy.HFClientTGI(model=model_id, port=port, url=url, max_tokens=TGI_MAX_TOKENS, stop=stop, temperature=temperature)
dspy.HFClientTGI(model='mistralai/Mistral-7B-Instruct-v0.2', port=1880, url='http://localhost', max_tokens=4096)
dspy.settings.configure(lm=vllm_llama2)

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
        # ChainOfThought generates answers using the configured local LLaMA LM via vLLM.
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
