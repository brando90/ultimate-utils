import json
def compare_answers(completions_file):
    # Initialize counter
    counter = 0
    total_questions = 0
    # Open and read the JSON file
    with open(completions_file, 'r') as file:
        completions = json.load(file)
    
    # Iterate over each dictionary in the list
    for completion in completions:
        total_questions += 1
        # Check if 'model_answer' equals 'math_gold_answer'
        if completion.get('model_answer') == completion.get('math_gold_answer'):
            counter += 1
    
    return counter, total_questions

# Call the function with the path to your completions.json file
counter,total_questions = compare_answers('/lfs/ampere1/0/ishanim/data/results_llama3_math/results_2024-m05-d18-t02h_52m_47s/completions.json')
print("Number of matching answers:", counter)
print("Total number of questions:", total_questions)
