import json
from pathlib import Path
from typing import Iterator, Optional, Union, Any
import os

import tenacity

from concurrent.futures import ProcessPoolExecutor, cpu_count

def get_iter_for_eval_data_set(path: Path, 
                               ) -> Iterator[dict]:
    """ Get an iterator for the evaluation data set. """
    path: Path = path.expanduser()
    if path.is_file() and path.suffix == '.json':
        with open(path, 'r') as file:
            data: list[dict] = json.load(file)
            return iter(data)
    elif 'GSM8K' in str(path):
        raise NotImplementedError
    elif 'OlympiadBench_Dataset' in str(path):
        return get_iter_multiple_files_with_multiple_data_points(path=path)
    elif 'Putnam_MATH_original_static2' in str(path):
        return get_iter_multiple_files_with_multiple_data_points(path=path)
    elif 'Putnam_MATH_original_static_final' in str(path):
        # ~/putnam-math/data/Putnam_MATH_original_static_final/Putnam_MATH_boxed_problems.json
        return get_iter_multiple_files_with_multiple_data_points(path=path)
    elif 'Putnam_MATH_variation_static2' in str(path):
        raise NotImplemented
    elif 'MATH' in str(path):
        # return get_iter_single_file_per_data_point(path=path)
        return process_files_multiprocessing(path=path)
    else:
        raise NotImplementedError
    
def get_iter_single_file_per_data_point(path : Path = Path('~/gold-ai-olympiad/data/MATH/test')
                                        ) -> Iterator[dict]:
    """ Get an iterator of single file per data point. e.g., when the MATH data set is stored as MATH/test/{category}/{problem_id}.json. """
    path: Path = path.expanduser()
    # recursively get all files and yield them
    for dirpath, dirnames, filenames in os.walk(path):
        # print(f'{dirpath=}')
        # print(f'{len(filenames)=}')
        for filename in filenames:
            if filename.endswith('.json'):
                file_path: Path = Path(dirpath, filename).expanduser()
                with open(file_path, 'r', encoding='utf-8') as file:
                    data: dict = json.load(file)
                    yield data

def process_files_multiprocessing(path: Path, max_workers: int = cpu_count()):
    """Recursively collects and processes JSON files with multiprocessing."""
    path = path.expanduser()

    def process_file(file_path: Path) -> dict:
        """Helper function to process each file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data: dict = json.load(file)
        return data

    def collect_files(path: Path):
        """Collect all JSON files recursively."""
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.json'):
                    yield Path(dirpath) / filename

    # Collect files and use multiprocessing to process them
    files = list(collect_files(path))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_file, files)

    for result in results:
        yield result

def get_iter_multiple_files_with_multiple_data_points(path: Path = Path('~/putnam-math/data/Putnam_MATH_original_static2/test'),
                                                      ) -> Iterator[dict]:
    """ Get an iterator to read multiple files with multiple data points. """
    path: Path = path.expanduser()
    # recursively get all files and for each file it has a json list of dicts. We want to yield each one
    for dirpath, dirnames, filenames in os.walk(path):
        # print(f'{dirpath=}')
        for filename in filenames:
            # print(f'{filename=}')
            if filename.endswith('.json'):
                file_path: Path = Path(dirpath, filename).expanduser()
                with open(file_path, 'r', encoding='utf-8') as file:
                    data: list[dict] = json.load(file)
                    for data_pt in data:
                        yield data_pt

def save_completions(path: Path, filename: str, completions: list[list[str]], model_answers: list[str], math_gold_probs_solns: list[dict], math_gold_answers: list[str]):
    # TODO: put an entry that says if model got it right or not
    print(f'-->saving completions for at {path=}')
    path: Path = path.expanduser()
    assert len(completions) == len(model_answers) == len(math_gold_probs_solns) == len(math_gold_answers), f'Length of completions, model_answers, math_gold_probs_solns, math_gold_answers should be equal but got: \n{len(completions)=}, {len(model_answers)=}, {len(math_gold_probs_solns)=}, {len(math_gold_answers)=} respectively.'
    results_vs_answers: list[dict] = []
    for i, (completion, model_answer, math_gold_probs_soln, math_gold_answer) in enumerate(zip(completions, model_answers, math_gold_probs_solns, math_gold_answers)):
        data: dict = {
            'completion': str(completion),  # TODO: maybe generalize to take list[list[CompletionOutput]] instead of list[list[str]
            'model_answer': model_answer,
            'math_gold_probs_soln': math_gold_probs_soln,
            # 'solution': math_gold_probs_soln['solution'],
            'math_gold_answer': math_gold_answer,
        }
        results_vs_answers.append(data)
    # save results as a list of dicts to a json file
    with open(path / filename, 'w', encoding='utf-8') as file:
        json.dump(results_vs_answers, file, indent=4)
    print(f'-->Done saving completions for at {path=}')

def putnam_2_jsonlines(
        path_2_src_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static2/', 
        output_dir: str = '~/putnam-math/data/',
        ) -> list[dict]:
    """ Converts a data set as dirs to json files with lists of json dicts to a single jsonlines file and returns the flattened list of dicts too. """
    # expand users
    path_2_src_dataset: Path = Path(path_2_src_dataset).expanduser()
    output_dir: Path = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # get all files from the src dataset, that dir has list of json files, each json file is a list of dicts, just concatenate them all and flatten
    all_data: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(path_2_src_dataset):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path: Path = Path(dirpath, filename).expanduser()
                with open(file_path, 'r', encoding='utf-8') as file:
                    data: list[dict] = json.load(file)
                    all_data.extend(data)

    # save all data to a jsonlines file, get final dir name from path_2_src_dataset and use that as the filename 
    final_dir_name: str = path_2_src_dataset.name
    output_path: Path = output_dir / f'{final_dir_name}.jsonl'
    with open(output_path, 'w', encoding='utf-8') as file:
        for data in all_data:
            json.dump(data, file)
            file.write('\n')
    return all_data

def dataset_in_folder_jsonfiles_list_dicts_2_jsonlines_file(
        path_2_src_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static2/',
        output_dir: str = '~/putnam-math/data/',
        ) -> list[dict]:
    """ Converts a data set as dirs to json files with lists of json dicts to a single jsonlines file and returns the flattened list of dicts too. """
    return putnam_2_jsonlines(path_2_src_dataset=path_2_src_dataset, output_dir=output_dir)

def olympiad_bench_2_hendrycks_math_format(path='~/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024'):
    path = Path(path).expanduser()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path: Path = Path(dirpath, filename).expanduser()
            with open(file_path, 'r', encoding='utf-8') as file:
                data: list[dict] = json.load(file)
                for data_pt in data:
                    from copy import copy
                    # data['original_solution'] = copy(data['solution'])
                    data_pt['original_solution'] = data_pt['solution']
                    data_pt['solution'] = ' '.join(data_pt['original_solution'])
                    assert data_pt['original_solution'] is not data_pt['solution'], 'Same strings! References are the same!'
            # now that we have the new data we need to create a new version of the data set
            file_path_new: Path = Path(f'{dirpath}_v2', filename).expanduser()
            Path(f'{dirpath}_v2').mkdir(exist_ok=True, parents=True)
            with open(file_path_new, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)

def get_model_answer_correct(path: str) -> None:
    from evals.utils import is_equiv_box_acc
    # load data
    path = Path(path).expanduser()
    with open(path, 'r', encoding='utf-8') as file:
        data: list[dict] = json.load(file)
    for data_completion in data:
        model_answer, gold_answer = data_completion['model_answer'], data_completion['math_gold_answer']
        correct: Union[str, bool] = is_equiv_box_acc(target_str=gold_answer, predicted_str=model_answer)  # False, True, 'Both_None'
        if correct == 'Both_None':
            raise ValueError('Fatal error, both None after we should have removed that in eval run')
        data_completion['correct'] = correct # store if it's correct, overwrite correct if it doesn't exit, fine to mutate in this case
    # save new data with correct per dict
    path_2_data_with_correct: Path = Path(path.parent / 'completions_with_correct.json') 
    with open(path_2_data_with_correct, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# -- Folder Folder Rec -> Jsonlines file

def dataset_in_folder_folder_rec_jsonfiles_2_jsonlines_file(
        # path_2_src_dataset: str = '~/putnam-math/data/MATH/test/',
        # output_path: str = '~/putnam-math/data/',
        path_2_src_dataset: str = '~/gold-ai-olympiad/data/MATH/test/',
        output_path: str = '~/gold-ai-olympiad/data/MATH/test.jsonl',
        convert_key_in_src_2_new_key_in_output: Optional[list[tuple[int, str]]] = None,  # [('hints', 'solution')]
        ) -> list[dict]:
    """ Converts data in folder to folders math categories to individual json files (are data points) to a single jsonlines file & returns list of dicts too. """
    # expand users
    path_2_src_dataset: Path = Path(path_2_src_dataset).expanduser()
    print(f'{path_2_src_dataset=}')
    output_path: Path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.is_dir() else output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{output_path=}')

    # get all json files into a list, the target dir has dir to dir for each math category Math/test/{cat}/{problem_id}.json which we walk recursively to get all json files and write them to a jsonlines file
    all_data: list[dict] = []
    all_filenames: list[str] = []
    all_filename_2_data: dict = dict()
    # all_dir_cat_filename_2_data: dict = dict()
    for dirpath, dirnames, filenames in os.walk(path_2_src_dataset):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path: Path = Path(dirpath, filename).expanduser()
                with open(file_path, 'r', encoding='utf-8') as file:
                    data: dict = json.load(file)
                    if convert_key_in_src_2_new_key_in_output:  # [], None are fasly
                        for src_key, new_key in convert_key_in_src_2_new_key_in_output:
                            data[new_key] = data[src_key]
                    all_data.append(data)
                    all_filenames.append(filename)
                    all_filename_2_data[filename] = data
                    # all_dir_cat_filename_2_data[dirpath][filename] = data  # need to initialize the inner dict for dirpath cat to not crash
    
    # save all data to a jsonlines file, get final dir name from path_2_src_dataset and use that as the filename
    # final_dir_name: str = f'{path_2_src_dataset.parts[-2]}_{ path_2_src_dataset.parts[-1]}'
    # output_path: Path = output_path / f'{final_dir_name}.jsonl'
    output_path: Path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.is_dir() else output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{output_path=}')
    with open(output_path, 'w', encoding='utf-8') as file:
        for data in all_data:
            json.dump(data, file)
            file.write('\n')
    return all_data

def seperate_proof_and_boxed_answer_problems_putnam_math():
    # TODO
    ...

# -- tests

def _test_create_putnam_jsonlines_file():
    lst: list[dict] = putnam_2_jsonlines()
    print(f'{len(lst)=}')

# -- Mains

def main_math_jsonlines_file():
    # NOTE: we want to have sample fields for all our data sets as the MATH data set. 

    # -- MATH -> jsonlines. fields are: problem, level, type, solution 
    # MATH/{split}/{cat}/{data_pt_id}.json -> MATH/{split}.jsonl
    path_2_src_dataset: str = '~/gold-ai-olympiad/data/MATH/test/'
    output_path: str = '~/gold-ai-olympiad/data/MATH/test.jsonl'
    lst: list[dict] = dataset_in_folder_folder_rec_jsonfiles_2_jsonlines_file(path_2_src_dataset, output_path)
    print(f'Number of jsonlines: {len(lst)=}, written to {output_path=}, from {path_2_src_dataset=}')
    
    # MATH/{split}/{cat}/{problem_id}.json -> MATH/{split}.jsonl
    path_2_src_dataset: str = '~/gold-ai-olympiad/data/MATH/train/'
    output_path: str = '~/gold-ai-olympiad/data/MATH/train.jsonl'
    lst: list[dict] = dataset_in_folder_folder_rec_jsonfiles_2_jsonlines_file(path_2_src_dataset, output_path)
    print(f'Number of jsonlines: {len(lst)=}, written to {output_path=}, from {path_2_src_dataset=}')

    # -- Khan Academy Math -> jsonlines. fields are: problem, hints (list[str]), we are going to make the hints a solution string (inspected a few in algebra and the answer is there although it's not boxed, so soln + answer there!)
    # amps/khan/{data_pt_id}.json -> amps/khan/{split}.jsonl
    # path_2_src_dataset: str = '~/gold-ai-olympiad/data/amps/khan/'
    path_2_src_dataset: str = '~/data/amps/khan/'
    output_path: str = '~/gold-ai-olympiad/data/amps/khan/train.jsonl'
    lst: list[dict] = dataset_in_folder_folder_rec_jsonfiles_2_jsonlines_file(path_2_src_dataset, output_path)
    print(f'Number of jsonlines: {len(lst)=}, written to {output_path=}, from {path_2_src_dataset=}')

    # -- TODO: mathematica, create jsonlines file, they have 5M, we don't need that much, a few from each dir is fine for our use case AF AIf
    # loop through each folder, get a 50 .txt files 
    # format
    """
Problem:
Consider the arithmetic sequence defined by $a_1=-\frac{43}{2}$, and $a_n=a_{n-1}+0$ for $n > 1$. Compute the nth partial sum, $S_n=\sum_{k=1}^n a_k$, where $n=26$.
Answer:
$-559$
    """
    # create a problem, solution, 
    # the solution create a solution using the name of the category + problem type (since that has the type of problem) + answer using an LLM (openai API, since we have a server running for this)
    # solution = llm.generate_completion(problem, answer, cat, prob_type, model)
    # Get the answer from the .txt file and put it as an boxed answer in the solution at the end by saying, The final answer is: \\boxed{answer}
    path_2_src_dataset: str = '/lfs/ampere1/0/brando9/data/amps/mathematica'
    output_path: str = '~/gold-ai-olympiad/data/amps/mathematica/train.jsonl'
    lst: list[dict] = dataset_in_folder_folder_rec_jsonfiles_2_jsonlines_file(path_2_src_dataset, output_path)
    print(f'Number of jsonlines: {len(lst)=}, written to {output_path=}, from {path_2_src_dataset=}')

if __name__ == '__main__':
    import time
    start = time.time()
    # _test_batch_data()
    # _test_create_putnam_jsonlines_file()
    # main_math_jsonlines_file()
    # olympiad_bench_2_hendrycks_math_format()
    get_model_answer_correct()
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")
    