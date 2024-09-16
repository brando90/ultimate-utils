import dspy

# Create an Example object
# qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")
qa_pair = dspy.Example(answer="This is an answer.")

# Specify the input field, leaving the answer as the output (label)
# qa_pair_with_input = qa_pair.with_labels("answer")

# print(qa_pair_with_input)
print(qa_pair)

#%%
# from dspy.datasets import HotPotQA

# # Load the dataset
# dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# # Tell DSPy that the 'question' field is the input and the 'answer' field is the label
# trainset = [x.with_inputs('question').with_labels('answer') for x in dataset.train]
# devset = [x.with_inputs('question').with_labels('answer') for x in dataset.dev]

# len(trainset), len(devset)
