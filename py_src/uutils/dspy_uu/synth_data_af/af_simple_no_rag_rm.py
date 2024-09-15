"""
ref: https://chatgpt.com/c/66e45ebd-26b0-8001-963d-0d2cc57e62cc

- extract prompt template
"""
from pathlib import Path
import dspy
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datasets import load_dataset

# from dspy.teleprompt import BootstrapFewShot
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from multiprocessing import cpu_count

import fire

class CrossEntropyStringMetric:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_cross_entropy(self, reference: str, prediction: str):
        """ Compute the cross-entropy loss between reference and prediction strings. """
        import torch
        import torch.nn.functional as F

        # Tokenize the reference and prediction strings
        ref_tokens = self.tokenizer(reference, return_tensors='pt')
        pred_tokens = self.tokenizer(prediction, return_tensors='pt')

        # Get the model's output (logits) for the prediction
        with torch.no_grad():
            outputs = self.model(**pred_tokens)
            logits = outputs.logits

        # Shift logits and labels for calculating cross-entropy loss
        # The labels should be the reference tokens
        labels = ref_tokens['input_ids'].to(logits.device)
        
        # Calculate cross-entropy loss per token
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        
        # Return the average cross-entropy loss for the sequence
        return loss.mean().item()

    def __call__(self, example, pred, trace=None):
        reference = example['formal_statement']  # Adjust based on dataset structure
        prediction = pred.formalization
        return self.compute_cross_entropy(reference, prediction)

# Define the DSPy Signature for formalizing math
class AutoFormalizeMath(dspy.Signature):
    """AutoFormalize (translate) natural language/informal mathematical statements to the formal Lean 4 programming language/interactive theorem prover"""
    informal_math_statement = dspy.InputField(desc="A natural language (informal) description of a mathematical statement")
    formalization = dspy.OutputField(desc="The autoformalization in the (formal) Lean 4 interactive theorem prover of the mathematical natural language statement")

# Define the Auto-Formalizer Module
class AutoFormalizer2Lean4(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_lean4_code = dspy.ChainOfThought(AutoFormalizeMath)
    
    def forward(self, informal_math_statement):
        # Generate the Lean formalization
        formalization = self.generate_lean4_code(informal_math_statement=informal_math_statement)
        # Return the result
        return dspy.Prediction(formalization=formalization)

def main(
    ds_trainset: str = 'hoskinson-center/proofnet'
    ):
    # Load the local HF model and tokenizer
    # export CUDA_VISIBLE_DEVICES=5
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    model_name = "meta-llama/Llama-2-7b-hf"  # Adjust model name as necessary
    # model_name = "gpt2"  # Adjust model name as necessary

    # Configure DSPy to use the local HF model as the LM.
    hf_lm = dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=hf_lm) 
    
    # Create dspy program to compile
    autoformalizer = AutoFormalizer2Lean4()

    # load dataset
    dataset = load_dataset(ds_trainset, split='validation')
    trainset = [dspy.Example(informal_math_statement=dpt['nl_statement'], formalization=dpt['formal_statement']).with_inputs('informal_math_statement') for dpt in dataset]
    eval_dataset = load_dataset(ds_trainset, split='test')

    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
    # The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=cpu_count())
    metric = CrossEntropyStringMetric(model=hf_lm.model, tokenizer=hf_lm.tokenizer)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, **config)
    complied_autoformalizer = teleprompter.compile(autoformalizer, trainset=trainset)

    # Test the pipeline with a new mathematical description
    description = "The sum of two odd numbers is even."
    pred = complied_autoformalizer(description)

    # Output the result
    print(f"Description: {description}")
    print(f"Formalized in Lean: {pred.formalization}")

    # from dspy.evaluate import Evaluate
    # evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)
    # evaluate_program(your_dspy_program)
    # Evaluate the DSPy model
    # per_device_eval_batch_size = 1
    # if output_dir_test is not None:
    #     output_dir_test: Path = Path(output_dir_test.format(date=formated_date)).expanduser()
    #     output_dir_test.mkdir(parents=True, exist_ok=True)
    #     eval_args = TrainingArguments(output_dir=output_dir_test, per_device_eval_batch_size=per_device_eval_batch_size, fp16=False, bf16=torch.cuda.is_bf16_supported(), report_to=report_to)
    #     trainer = Trainer(model=hf_lm.model, args=eval_args, train_dataset=None, eval_dataset=eval_dataset)
    #     # results: dict[str, float] = trainer.evaluate(test_dataset)
    #     results: dict[str, float] = eval_hf(trainer, name='', path=path, split='test')

if __name__ == "__main__":
    import time
    start_time = time.time()
    fire.Fire(main)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
