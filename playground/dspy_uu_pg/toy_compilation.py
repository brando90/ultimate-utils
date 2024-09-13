import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate

# Configure DSPy with the desired LM (GPT-3.5-turbo) and no retrieval model in this case.
# We're focusing on few-shot generation without retrieval augmentation here.
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo, rm=None)

# Step 1: Define a small, high-quality hardcoded dataset (3-5 examples)
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

# Convert the dataset into DSPy examples with input/output fields
trainset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in train_data]
devset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in dev_data]

# Step 2: Define the RAG program (Simple QA Generation)
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought generates answers using the configured LM (GPT-3.5-turbo).
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        # Pass the question through the LM to generate an answer.
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)

# Step 3: Metric to evaluate exact match between predicted and expected answer.
def exact_match_metric(example, pred, trace=None):
    return example['answer'].lower() == pred.answer.lower()

# Step 4: Use teleprompter (BootstrapFewShot) to optimize few-shot examples for the best performance.
# It optimizes the examples selected from the train set based on the exact match metric.
teleprompter = BootstrapFewShot(metric=exact_match_metric)

# Compile the SimpleQA program with optimized few-shots from the train set.
compiled_simple_qa = teleprompter.compile(SimpleQA(), trainset=trainset)

# Step 5: Test with a sample question and evaluate the performance
my_question = "What is the capital of Japan?"
pred = compiled_simple_qa(my_question)

# Output the predicted answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

# Evaluate the compiled program on the dev set using the exact match metric.
evaluate_on_dev = Evaluate(devset=devset, num_threads=1, display_progress=False)
evaluation_score = evaluate_on_dev(compiled_simple_qa, metric=exact_match_metric)

print(f"Evaluation Score on Dev Set: {evaluation_score}")
