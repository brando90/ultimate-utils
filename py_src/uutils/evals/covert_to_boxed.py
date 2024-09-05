import json

# Load the JSON file
with open('/lfs/ampere1/0/ishanim/massive-evaporation-4-math-new/data/OlympiadBench_Dataset/data_math/TP_TO_maths_en_COMP.json', 'r') as file:
    data = json.load(file)

# Modify each dictionary in the list
for item in data:
    final_answer_str = f"The final answer is \\boxed{{{item['final_answer'][0]}}}"
    item['solution'].append(final_answer_str)

# Save the modified data to a new JSON file
with open('/lfs/ampere1/0/ishanim/massive-evaporation-4-math-new/data/OlympiadBench_Dataset/data_math/TPpython covert_to_boxed.py_TO_maths_en_COMP_modified.json', 'w') as file:
    json.dump(data, file, indent=4)

print("The new dataset has been saved to dataset_2.json")
