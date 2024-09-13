import dspy
import json

# Step 1: Define the Task Signature
# The synthetic data we want to generate will be a question-answer pair.
class GenerateSyntheticData(dspy.Signature):
    """This signature defines the input-output pair for synthetic data generation."""
    
    # Input: the context/topic about which we want to generate questions and answers
    context = dspy.InputField(desc="The general topic or subject area, e.g., 'science', 'history', etc.")
    
    # Output: synthetic question and answer pair
    question = dspy.OutputField(desc="A question related to the context")
    answer = dspy.OutputField(desc="The corresponding answer to the question")

# Step 2: Define the Synthetic Data Generation Module
class SyntheticDataGenerator(dspy.Module):
    """This module generates synthetic data: question-answer pairs given a context."""
    
    def __init__(self):
        super().__init__()
        
        # Chain of thought to generate the question-answer pair.
        self.generate_qa_pair = dspy.ChainOfThought(GenerateSyntheticData)
    
    def forward(self, context):
        """Takes a context and generates a synthetic question-answer pair."""
        
        # Generate the question and answer based on the context
        prediction = self.generate_qa_pair(context=context)
        
        # Return the generated synthetic question-answer pair
        return dspy.Prediction(question=prediction.question, answer=prediction.answer)

# Step 3: Compile the Module
from dspy.teleprompt import BootstrapFewShot

# No specific validation needed in this case, since we are just generating synthetic data.
# So, we'll directly compile the module without specific validation logic.
teleprompter = BootstrapFewShot()

# Compile the synthetic data generation program
compiled_generator = teleprompter.compile(SyntheticDataGenerator())

# Step 4: Generate Synthetic Data
# Define some topics/contexts that we want to generate synthetic questions and answers for.
contexts = [
    "the history of the Roman Empire", 
    "basic principles of quantum mechanics", 
    "the process of photosynthesis", 
    "climate change and its impact", 
    "the American Civil War"
]

# Initialize a list to store the synthetic data
synthetic_data = []

# Loop over each context and generate synthetic question-answer pairs
for context in contexts:
    # Generate synthetic data for this context
    pred = compiled_generator(context)
    
    # Store the generated data in a structured format (e.g., dictionary)
    synthetic_data.append({
        "context": context,
        "question": pred.question,
        "answer": pred.answer
    })

# Step 5: Save the Synthetic Data
# Save the generated synthetic data to a JSON file for later fine-tuning.
with open("synthetic_qa_data.json", "w") as f:
    json.dump(synthetic_data, f, indent=4)

# Print the generated synthetic data for inspection
for entry in synthetic_data:
    print(f"Context: {entry['context']}")
    print(f"Question: {entry['question']}")
    print(f"Answer: {entry['answer']}")
    print("-" * 50)