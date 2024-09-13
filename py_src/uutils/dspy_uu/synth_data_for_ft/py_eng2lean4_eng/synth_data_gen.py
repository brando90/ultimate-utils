"""
ref: https://chatgpt.com/share/66e4c1cd-b428-8001-8ea4-9998d3bd962c

TODO: lots of things to fix...
"""
import dspy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from uutils.dspy_uu.hf_models import load_mdl_and_tok
import torch

# Step 1: Synthetic Data Generation (Python -> Lean4)
class Python2Lean(dspy.Signature):
    """Generates Lean4 code and docstring from Python code and docstring."""
    python_code = dspy.InputField(desc="Python code with docstring")
    lean_code = dspy.OutputField(desc="Generated Lean4 code")
    lean_docstring = dspy.OutputField(desc="Generated Lean4 docstring")

class SyntheticDataGen(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_translation = dspy.ChainOfThought(Python2Lean)

    def forward(self, python_code):
        # Generate synthetic Lean4 code and docstring
        prediction = self.generate_translation(python_code=python_code)
        return dspy.Prediction(lean_code=prediction.lean_code, lean_docstring=prediction.lean_docstring)

# Step 2: Fine-tuning the model on Synthetic Data
class FineTuneModule(dspy.Module):
    def __init__(self, pretrained_model_name_or_path, training_args):
        super().__init__()
        self.model, self.tokenizer = load_mdl_and_tok(pretrained_model_name_or_path)
        self.training_args = training_args
        self.trainer = None

    def wrap_synth_data_to_hf_dataset(self, synthetic_data):
        # Create a Hugging Face dataset from synthetic data for fine-tuning
        data = {
            'input_ids': [self.tokenizer.encode(synthetic_data.lean_docstring)],
            'labels': [self.tokenizer.encode(synthetic_data.lean_code)]
        }
        dataset = Dataset.from_dict(data)
        return dataset

    def forward(self, synthetic_data, eval_dataset=None):
        # Wrap the synthetic data into a Hugging Face dataset
        train_dataset = self.wrap_synth_data_to_hf_dataset(synthetic_data)

        # Set up the Trainer for fine-tuning the model
        if self.trainer is None:
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
        # Fine-tune the model
        self.trainer.train()

        return self.model  # Return the fine-tuned model

# Step 3: Integrate the steps into a full AutoFormalization pipeline
class AutoFormalizationPipeline(dspy.Module):
    def __init__(self, pretrained_model_name_or_path, training_args):
        super().__init__()
        self.synthetic_data_gen = SyntheticDataGen()
        self.fine_tune_module = FineTuneModule(pretrained_model_name_or_path, training_args)

    def forward(self, python_code, eval_dataset=None):
        # Step 1: Generate synthetic Lean4 code and docstring
        synth_data = self.synthetic_data_gen(python_code=python_code)

        # Step 2: Fine-tune the model on the synthetic data
        model = self.fine_tune_module(synthetic_data=synth_data, eval_dataset=eval_dataset)

        return dspy.Prediction(synth_data=synth_data, fine_tuned_model=model)

# Step 4: Set up the evaluation using ProofNet and MiPro optimizer

class ProofNetEvaluator:
    def __init__(self):
        self.proofnet_data = load_dataset("hoskinson-center/proofnet", "validation")

    def evaluate(self, model):
        correct = 0
        total = len(self.proofnet_data['validation'])
        for example in self.proofnet_data['validation']:
            nl_statement = example['nl_statement']
            formal_statement = example['formal_statement']
            input_ids = model.tokenizer.encode(nl_statement, return_tensors='pt')
            output_ids = model.model.generate(input_ids)
            generated_formal = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if generated_formal.strip() == formal_statement.strip():
                correct += 1
        return correct / total

# Example usage
if __name__ == "__main__":
    # Define training arguments for the Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500
    )

    # Initialize the pipeline
    pipeline = AutoFormalizationPipeline(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",  # Example model
        training_args=training_args
    )

    # Example Python code with docstring to be translated into Lean4
    python_code = """def factorial(n: int) -> int:
        '''Returns the factorial of a number.'''
        if n == 1:
            return 1
        return n * factorial(n-1)
    """

    # Load ProofNet for evaluation
    proofnet_evaluator = ProofNetEvaluator()

    # Run the pipeline
    pred = pipeline(python_code, eval_dataset=proofnet_evaluator.proofnet_data)

    # Evaluate the fine-tuned model using ProofNet
    accuracy = proofnet_evaluator.evaluate(pred.fine_tuned_model)
    print(f"Accuracy on ProofNet autoformalization task: {accuracy * 100:.2f}%")