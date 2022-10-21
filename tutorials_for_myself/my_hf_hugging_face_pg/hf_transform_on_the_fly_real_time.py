"""

ref:
- todo, find hf post I discussed...
- vit has a comment on this: https://huggingface.co/blog/fine-tune-vit


"""

#%%
"""
ref: https://huggingface.co/blog/fine-tune-vit

Processing the Dataset
...

First, though, you'll need to update the last function to accept a batch of data, as that's what ds.with_transform expects.

ds = load_dataset('beans')

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs

You can directly apply this to the dataset using ds.with_transform(transform).

prepared_ds = ds.with_transform(transform)

Now, whenever you get an example from the dataset, the transform will be applied in real time (on both samples and slices, as shown below)

prepared_ds['train'][0:2]

This time, the resulting pixel_values tensor will have shape (2, 3, 224, 224).
"""