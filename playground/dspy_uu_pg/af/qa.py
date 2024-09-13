import dspy
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 3: Configure the DSPy environment with the LM and RM
llama_model = HuggingFaceLM()
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=llama_model, rm=colbertv2_wiki17_abstracts)

# Step 4: Define the DSPy Signature for generating answers
class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="Relevant facts or knowledge")
    question = dspy.InputField(desc="The question to be answered")
    answer = dspy.OutputField(desc="A short fact-based answer")

# Step 5: Define the RAG Module for the question-answering task
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        # Step 1: Retrieve relevant context (e.g., passages of knowledge)
        context = self.retrieve(question).passages
        
        # Step 2: Use the context to generate an answer
        prediction = self.generate_answer(context=context, question=question)
        
        return dspy.Prediction(context=context, answer=prediction.answer)

# Step 6: Set up the teleprompter and optimizer (BootstrapFewShot)
from dspy.teleprompt import BootstrapFewShot

# Validation function that checks if the predicted answer is correct
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Compile the RAG model with few-shot optimization
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Define a dataset (e.g., HotPotQA)
from dspy.datasets import HotPotQA
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Train and development sets
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# Compile the RAG program with few-shot optimization
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

# Step 7: Test the pipeline with a new question
question = "Who wrote the Declaration of Independence?"
pred = compiled_rag(question)

# Output the result
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts: {pred.context}")
