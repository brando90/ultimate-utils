#%%

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]