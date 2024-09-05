"""
Claude refactored original MetaMath code: https://claude.ai/chat/59b08651-0841-4086-a0b6-3bd4935a80bf
GPT4: https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/9641763f-f439-4f19-ad74-6e666b0fbd8a
"""
from pathlib import Path
import os
import json
from typing import Callable, Optional, Union, Tuple, List, Any

import sys

def remove_boxed(string: str) -> Union[str, None]:
    """
    Removes the LaTeX "\\boxed{}" command from a given string, if it's not a boxed string else stops with an error.

    This function checks if the input string starts with "\\boxed{" and ends with "}".
    If both conditions are met, it returns the substring between these delimiters.
    If the input string is not in the expected format, the function returns None.

    Parameters:
    s (str): The input string to process.

    Returns:
    str or None: The substring with the "\\boxed{}" command removed, or None if the input string is not in the expected format.
    """
    if string is None:
        return None
    left = "\\boxed{"
    
    try:
        # Check if the string starts with "\\boxed{"
        assert string[:len(left)] == left
        # Check if the string ends with "}"
        assert string[-1] == "}"
        
        # If both conditions are met, return the substring between the delimiters
        return string[len(left):-1]
    except AssertionError:
        # If the string is not in the expected format, return None
        return None

def last_boxed_only_string(string: str) -> Union[str, None]:
    """
    Extracts the last LaTeX boxed expression str from a given string. It does extract the LaTeX command e.g.,

        Option 2: "...\\boxed{A}..." → "\\boxed{A}" 

    This function finds the last occurrence of either "\\boxed" or "\\fbox" in the input string,
    and returns the substring that starts with this command and ends with the corresponding
    closing brace. If no boxed expression is found, the function returns None.

    Parameters:
    string (str): The input string to search for a boxed expression.

    Returns:
    str or None: The last boxed expression found in the string, or None if no boxed expression is found.

    ref: 
        - https://claude.ai/chat/1b706c36-9776-4829-97a6-f7056b4d9c21
        - https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/dae475f1-47b4-45ff-b7b0-471a100d2312
    """
    # Find the index of the last occurrence of "\\boxed", starting at the beginning of the match
    idx = string.rfind("\\boxed")

    # If "\\boxed" is not found, try to find the last occurrence of "\\fbox"
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None  # Index of the closing brace (inclusive)
    num_left_braces_open = 0

    # Iterate through the string starting from the found index, regex cannot solve matching parens due to not having memory, ref: https://claude.ai/chat/1b706c36-9776-4829-97a6-f7056b4d9c21
    while i < len(string):
        # Count the number of opening braces
        if string[i] == "{":
            num_left_braces_open += 1
        # Count the number of closing braces
        if string[i] == "}":
            num_left_braces_open -= 1
            # If all opened braces are closed, mark the index of the closing brace
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    # If no closing brace is found, return None
    if right_brace_idx is None:
        retval = None
    # Otherwise, return the substring between the command and the closing brace
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def batch_data(data_list: list[str], batch_size: int = 1) -> list[list[Any]]:
    """
    Splits the input data_list into smaller batches of specified batch_size.

    Args:
        data_list (list): The list of items to be batched.
        batch_size (int): The number of items in each batch.

    Returns:
        list: A list of lists, where each inner list is a batch containing up to batch_size items.
    
    Each batch contains exactly batch_size items except potentially the last batch,
    which may contain fewer items if the length of data_list is not a multiple of batch_size.

    Ref: from math_eval.py from MetaMath https://github.com/meta-math/MetaMath/blob/main/eval_math.py
    """
    # Calculate the number of full batches
    n = len(data_list) // batch_size
    batch_data = []
    
    # Append each batch except the last to the batch_data list
    for i in range(n-1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    # Handle the last batch separately to include all remaining items because the case when the number of remaining elements is less than batch size
    last_start = (n - 1) * batch_size
    batch_data.append(data_list[last_start:])
    return batch_data

# -- What was in util.py in MetaMath https://github.com/meta-math/MetaMath/blob/main/util.py, comments with help from Claude: https://claude.ai/chat/59b08651-0841-4086-a0b6-3bd4935a80bf

def only_until_first_boxed_from_tokens(string: str, tokens: List[str]) -> Optional[List[str]]:
    """
    Extract the tokens until the first boxed content in the given string.

    Args:
        string (str): The input string.
        tokens (List[str]): The list of tokens.

    Returns:
        Optional[List[str]]: The tokens until the first boxed content, or None if no boxed content is found.
    """
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]

def clean_numbers(sample: Optional[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
    """
    Clean numbers in the given sample.

    Args:
        sample (Optional[Tuple[str, ...]]): A tuple of strings representing the sample.

    Returns:
        Optional[Tuple[str, ...]]: A tuple of strings with cleaned numbers, or None if the input sample is None.
    """
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string: str) -> str:
    """
    Clean numbers in the given string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with cleaned numbers.
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def fix_fracs(string: str) -> str:
    """
    Fix fractions in the given string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with fixed fractions.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string: str) -> str:
    """
    Fix fractions of the form "a/b" in the given string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with fixed fractions.
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string: str) -> str:
    """
    Remove units on the right side of the given string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with units removed.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string: str) -> str:
    """
    Fix square root expressions in the given string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with fixed square root expressions.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string: str) -> str:
    """
    Strip the given string by removing unwanted characters and formatting.

    This function performs various string manipulations to clean up the input string.
    It removes linebreaks, inverse spaces, replaces double backslashes with a single backslash,
    replaces 'tfrac' and 'dfrac' with 'frac', removes '\\left' and '\\right', removes degree symbols,
    removes dollar signs, removes units on the right side, removes percentage symbols,
    adds a leading '0' before a decimal point if necessary, removes short variable assignments,
    fixes square root expressions, removes spaces, fixes fractions, and replaces '0.5' with '\\frac{1}{2}'.

    TODO: why doesn't this use sympy?
    """
    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")

    # Replace double backslashes with a single backslash
    string = string.replace("\\\\", "\\")

    # Replace 'tfrac' and 'dfrac' with 'frac'
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove '\\left' and '\\right'
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove degree symbols
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("\\$", "")

    # Remove units on the right side
    string = remove_right_units(string)

    # Remove percentage symbols
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # Add a leading '0' before a decimal point if necessary
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # Remove short variable assignments (e.g., "k = " or "q = ")
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Fix square root expressions (e.g., "sqrt3" to "sqrt{3}")
    string = fix_sqrt(string)

    # Remove spaces
    string = string.replace(" ", "")

    # Fix fractions (e.g., "\frac1b" to "\frac{1}{b}")
    string = fix_fracs(string)

    # Replace "0.5" with "\frac{1}{2}"
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Fix fractions of the form "X/Y" to "\frac{X}{Y}"
    string = fix_a_slash_b(string)

    return string

def is_equiv(str1: Optional[str], str2: Optional[str], verbose: bool = False) -> Union[str, bool]:
    """
    Check if two strings are equivalent after stripping.
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return None  
    # if either of them is none (after already checking that both were none)
    if str1 is None or str2 is None:  # case: 1 0, 0 1 -> not equal strings
        return False
    try:
        # Strip the given strings by removing unwanted characters and formatting, are they equal now after striping/cleaning?
        ss1: str = strip_string(str1)
        ss2: str = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
    
def is_equiv_box_acc(target_str: Optional[str], predicted_str: Optional[str], verbose: bool = False) -> Union[str, bool]:
    """
    Check if target answer string is equivalent to predicted string after stripping. 

    old: process_results from meta math
    """
    if target_str is None:
        print(f'--> Warning: {target_str=} so likely question is a pure reasoning question')
        return 'Target_None'  # no need to look at model pred since we are only computing boxed accuracy of answers
    # if it's not a reasoning question, then you want to compare the target and predicted answers (while predicted could be None if model failed to generate an answer)
    # invariant: target_str is not None
    assert target_str is not None, f'{target_str=} should not be None'
    assert isinstance(target_str, str), f'{target_str=} should be a string'
    if predicted_str is None:
        return False  # since target is not None and predicted is None it's not a match
    try:
        # Strip the given strings by removing unwanted characters and formatting, are they equal now after striping/cleaning?
        ss1: str = strip_string(target_str)
        ss2: str = strip_string(predicted_str)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception: return target_str == predicted_str

# class NotEqual:
#     """
#     A class that always returns False when compared for equality.

#     This class is used as a sentinel value to represent inequality.
#     When an instance of this class is compared with any other object using the '==' operator,
#     it always returns False.
#     """

#     def __eq__(self, other: Any) -> bool:
#         """
#         Compare the NotEqual instance with another object for equality.

#         Args:
#             other (Any): The object to compare with.

#         Returns:
#             bool: Always returns False.
#         """
#         return False

# -- Boxed Accuracy

def extract_model_answers(completions: list, extract_answer_func: Callable) -> list[Union[str, None]]:
    assert len(completions) > 0, f'Expected completions to be a list of completions but got: {completions=}'
    # do we have a single completion per prompt? list[str] 
    if isinstance(completions[0], str):
        model_answers: list[str] = [extract_answer_func(completion) for completion in completions]
    # do we have multiple completions per prompt? list[list[str]]
    elif isinstance(completions[0], list) and isinstance(completions[0][0], str):
        model_answers: list[str] = [extract_answer_func(completions_per_prompt) for completions_per_prompt in completions]  # completions_per_prompt since it can be multiple completions per prompt (completions for completions)
    else: 
        raise ValueError(f'Expected completions to be a list of completion (e.g., str or list completions) for each prompt but got: {completions=}')
    return model_answers

def extract_gold_answers(math_gold_probs_solns: list[dict]) -> list[str]:
    math_gold_answers: list[str] = [remove_boxed(last_boxed_only_string(data["solution"])) for data in math_gold_probs_solns]
    return math_gold_answers

def eval_boxed_accuracy_results(math_gold_answers: list[str], 
                                model_answers: list[str], 
                                verbose_eval: bool = False, 
                                ) -> dict:
    """ Evaluate based on boxed accuracy. Return list of which data points where correct. See return statement for info about output. """
    assert len(model_answers) == len(math_gold_answers), f'Number of model answers should match number of gold answers but got: {len(model_answers)=} and {len(math_gold_answers)=} respectively.'
    # - Compute results per data point
    results: list[Union[str, bool]] = []
    for model_answer, gold_answer in zip(model_answers, math_gold_answers):
        correct: Union[str, bool] = is_equiv_box_acc(target_str=gold_answer, predicted_str=model_answer)  # False, True, 'Both_None'
        results.append(correct)
    # filter out things that aren't True or False (only box answers allowed)
    results_boxed = [result for result in results if result != 'Target_None']
    # - Compute boxed accuracy
    boxed_acc: float = sum(results_boxed) / len(results_boxed)
    # - Prepare results, print & return them
    results_d: dict = {'boxed_acc': boxed_acc, 'len(results)': len(results), 'len(results_boxed)': len(results_boxed), 'sum(results_boxed)': sum(results_boxed), 'results': results, 'results_boxed': results_boxed}
    if verbose_eval:
        print(f'{results_d["boxed_acc"]=} \n {results_d["len(results)"]=} \n {results_d["len(results_boxed)"]=} \n {results_d["sum(results_boxed)"]=}')
    return results_d

def collect_invalid_outputs(math_prompts: list[str],  gold_answers: list[str], completions: list[str], model_answers: list[str], invalid_outputs: list[dict] = []):
    assert NotImplementedError, 'Not implemented yet.'
    # invalid_outputs.append({'question': math_prompt, 'output': completion, 'gold_answer': gold_answer, 'model_answer': model_answer})
    # return invalid_outputs

# -- Misc

def get_dtype_for_vllm(dtype: Optional[str] = None):
    """ Return half precision since it's inference either bfloat16 or float16, or defualt to user input. """
    if dtype == 'auto':
        return 'auto'  # defaults to config file according to llm.py
    elif dtype == 'float32':
        return 'float32'
    else:
        import torch
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():  # Check for Apple Silicon (M1/M2)
                device = torch.device("mps")  # Use the Metal Performance Shaders (MPS) on Mac M1/M2
                dtype = torch.float32  # MPS typically uses float32
            else:
                device = torch.device("cpu")  # Default to CPU
                dtype = torch.float32
        else:
            if torch.cuda.is_bf16_supported():
                device = torch.device("cuda:0")  # Use bf16 on supported GPUs
                dtype = torch.bfloat16
            else:
                device = torch.device("cuda:0")  # Use fp16 on other GPUs
                dtype = torch.float16
        return dtype

def load_model(pretrained_model_name_or_path, verbose: bool = False, max_length: int = 1024):
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
    # TODO: improve training by not using max_length suff
    if pretrained_model_name_or_path == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, max_length=1024)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        # model.resize_token_embeddings(len(tokenizer))  # leaving for reference, not needed since pad = eos for us
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        max_length: int = tokenizer.model_max_length
        print(f'{max_length=}')
    elif 'Llama-2' in pretrained_model_name_or_path or 'Mistral' in pretrained_model_name_or_path:
        # - LLama2, later qlora: https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L347C13-L347C13
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        if hasattr(model.config, "context_length"): 
            print("Context length:", model.config.context_length)
        else:
            max_length = 4096
        max_length: int = 4096
        tokenizer.model_max_length = max_length
        print(f'{max_length=}')
    elif 'Llama-2' in pretrained_model_name_or_path or 'Mistral-7B-Instruct-v0.2' in pretrained_model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        # v0.1 vs v.02 context size 32k context window (vs 8k context in v0.1)
        tokenizer.model_max_length = max_length  # TODO: check if this is correct
        print(f'{max_length=}')
    elif 'gemma' in pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="auto", torch_dtype=torch_dtype)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token  
        print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        # get context length for setting max length for training
        if hasattr(model.config, "context_length"):
            # seems gemma model doesn't have this available issue: https://huggingface.co/google/gemma-2b/discussions/32
            print("Context length:", model.config.context_length)
            max_length = model.config.context_length
        else:
            print(f'{max_length=}')
        max_length: int = max_length
        print(f'{max_length=}')
    else:
        torch.cuda.empty_cache() # Clear CUDA cache to free up memory
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", trust_remote_code=True)
        print(f'{tokenizer=}')
        print(f'{tokenizer.pad_token_id=} {tokenizer.eos_token_id=}')
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        print(f'{tokenizer.pad_token_id=} {tokenizer.eos_token_id=}')
        # get context length for setting max length for training
        print(f'{model.config=}')
        if max_length is None:
            if hasattr(model.config, "context_length"):
                max_length: int = model.config.context_length 
                print("Context length:", model.config.context_length)
            else:
                max_length: int = 1024
        else:
            print(f"Context length not found in model.config, so using your default or hardcoded value. Model is {pretrained_model_name_or_path=}.")
            # max_length: int = 4  # for debugging
            max_length: int = max_length  # for debugging
            # max_length: int = 128_000  # ref: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
        model_weight_norm = sum([torch.norm(param, p=2).item() for param in model.parameters()])
        print(f'{device=} Model device: {next(model.parameters()).device}')
        print("Number of parameters:", sum(p.numel() for p in model.parameters()))
        print(f'{model_weight_norm=}')
    if verbose:
        print(model, tokenizer)
    return model, tokenizer

# -- tests

def _test_get_final_answer_string_only(
        path_2_MATH: str = '~/gold-ai-olympiad/data/MATH/train',
        ):
    """ """
    path_2_MATH: Path = Path(path_2_MATH).expanduser()
    for dirpath, dirnames, filenames in os.walk(path_2_MATH):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path: Path = Path(dirpath, filename).expanduser()
                with open(file_path, 'r', encoding='utf-8') as file:
                    data: dict = json.load(file)
                    # problem: str = data["problem"]
                    solution: str = data["solution"]
                    boxed_answer: str = last_boxed_only_string(solution)
                    answer: str = remove_boxed(boxed_answer)
                    print(f'{boxed_answer=}')
                    print(f'{answer=}')
                    print()

def _test_last_boxed_open_string_return():
    test_string = "Here is an equation \\boxed{E=mc^2} and some more text."
    extracted_val: Union[str, None] = last_boxed_only_string(test_string)
    assert extracted_val == "\\boxed{E=mc^2}", f"Expected '\\boxed{{E=mc^2}}', but got '{extracted_val}'"

    test_string = r"Here is an equation \boxed{E=mc^2} and some more text."
    extracted_val: Union[str, None] = last_boxed_only_string(test_string)
    assert extracted_val == "\\boxed{E=mc^2}", f"Expected '\\boxed{{E=mc^2}}', but got '{extracted_val}'"

    test_string = r"Here is an equation \boxed{eq1} and some more text \boxed{eq2} another equation \boxed{eq3}"
    extracted_val: Union[str, None] = last_boxed_only_string(test_string)
    assert extracted_val == "\\boxed{eq3}", f"Expected '\\boxed{{eq3}}', but got '{extracted_val}'"


def _test_batch_data():
    # Example data_list containing strings
    data_list = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    
    # Test case 1: Batch size of 1
    batch_size_1 = batch_data(data_list, batch_size=1)
    print("Batch size 1:", batch_size_1)
    
    # Test case 2: Batch size of 3
    batch_size_3 = batch_data(data_list, batch_size=3)
    print("Batch size 3:", batch_size_3)
    
    # Test case 3: Batch size equal to the length of the data list
    batch_size_full = batch_data(data_list, batch_size=len(data_list))
    print("Batch size full (equal to list length):", batch_size_full)

def _test_eval_boxed_accuracy_results():
    # All correct
    math_gold_answers: list[str] = ['A', 'B', 'C', 'D', 'E']
    model_answers: list[str] = ['A', 'B', 'C', 'D', 'E']
    results_d: dict = eval_boxed_accuracy_results(math_gold_answers, model_answers, verbose_eval=True)
    print(f'\n{results_d=}')
    assert results_d['boxed_acc'] == 1.0, f'{results_d["boxed_acc"]=} should be 1.0'

    # First two problems are pure reasoning, so we have two None's in the gold answers, all correct
    math_gold_answers: list[str] = [None, None, 'C', 'D', 'E']
    model_answers: list[str] = ['A', 'B', 'C', 'D', 'E']
    results_d: dict = eval_boxed_accuracy_results(math_gold_answers, model_answers, verbose_eval=True)
    print(f'\n{results_d=}')
    assert results_d['boxed_acc'] == 1.0, f'{results_d["boxed_acc"]=} should be 1.0'

    # First two problems are pure reasoning, so we have two None's in the gold answers but two mistakes, 1/3 correct
    math_gold_answers: list[str] = [None, None, 'C', 'D', 'E']
    model_answers: list[str] = ['A', 'B', 'X', 'Y', 'E']
    results_d: dict = eval_boxed_accuracy_results(math_gold_answers, model_answers, verbose_eval=True)
    print(f'\n{results_d=}')
    assert results_d['boxed_acc'] == 1/3, f'{results_d["boxed_acc"]=} should be 1/3'
    
if __name__ == '__main__':
    import time
    start = time.time()
    # _test_get_final_answer_string_only()
    # _test_last_boxed_open_string_return()
    # _test_batch_data()
    _test_eval_boxed_accuracy_results()
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")
    