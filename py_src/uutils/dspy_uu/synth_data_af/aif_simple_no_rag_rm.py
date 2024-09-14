"""
ref: https://chatgpt.com/c/66e45ebd-26b0-8001-963d-0d2cc57e62cc

- extract prompt template
"""
import dspy
from transformers import AutoModelForCausalLM, AutoTokenizer
from colbert.infra import ColBERT
import os
from datasets import load_dataset

from multiprocessing import cpu_count

import fire

# Load the local HF model and tokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = 5
model_name = "meta-llama/Llama-2-7b-hf"  # Adjust model name as necessary
# model_name = "gpt2"  # Adjust model name as necessary

#  Configure DSPy to use the local HF model as the LM.
huggingface_lm = dspy.HFModel(model=model_name)
dspy.settings.configure(lm=huggingface_lm) 

# Step 4: Define the DSPy Signature for formalizing math
class AutoFormalizeMath(dspy.Signature):
    """AutoFormalize (translate) natural language/informal mathematical statements to the formal Lean 4 programming language/interactive theorem prover"""
    description = dspy.InputField(desc="A natural language (informal) description of a mathematical statement")
    formalization = dspy.OutputField(desc="The autoformalization in the (formal) Lean 4 interactive theorem prover of the mathematical natural language statement")

# Step 5: Define the Auto-Formalizer Module
class AutoFormalizer2Lean4(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_lean4_code = dspy.ChainOfThought("informal_nl_math -> formal_lean4_code")
    
    def forward(self, informal_nl_math):
        # Generate the Lean formalization
        formal_lean4_code = self.generate_lean4_code(informal_nl_math=informal_nl_math)
        # Return the result
        return dspy.Prediction(formalization=formal_lean4_code)

def main(
    ds_trainset: str = 'hoskinson-center/proofnet'
    ):
    # Set up optimizer/teleprompter with few-shot optimization
    from dspy.teleprompt import BootstrapFewShot
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch
    
    # Create dspy program to compile
    autoformalizer = AutoFormalizer2Lean4()

    # load dataset
    trainset = load_dataset(ds_trainset, split='validation')

    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
    # The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=cpu_count())
    teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
    complied_autoformalizer = teleprompter.compile(autoformalizer, trainset=trainset)

    # Test the pipeline with a new mathematical description
    description = "The sum of two odd numbers is even."
    pred = complied_autoformalizer(description)

    # Output the result
    print(f"Description: {description}")
    print(f"Formalized in Lean: {pred.formalization}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    fire.Fire(main)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
