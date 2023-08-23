def get_hardcoded_full_mds_num_classes(sources: list[str] = ['hardcodedmds']) -> int:
    """
    todo: this is a hack, fix it
    - use the dataset_spec in each data set
```
  "classes_per_split": {
    "TRAIN": 140,
    "VALID": 30,
    "TEST": 30
```
    but for imagenet do something else.
    argh, idk what this means
```
{
  "name": "ilsvrc_2012",
  "split_subgraphs": {
    "TRAIN": [
      {
        "wn_id": "n00001740",
        "words": "entity",
        "children_ids": [
          "n00001930",
          "n00002137"
        ],
        "parents_ids": []
      },
      {
        "wn_id": "n00001930",
        "words": "physical entity",
        "children_ids": [
          "n00002684",
          "n00007347",
          "n00020827"
        ],
```
count the leafs in train & make sure it matches 712 158 130 for each split

ref: https://github.com/google-research/meta-dataset#dataset-summary
    """
    print(f'{sources=}')
    n_cls: int = 0
    if sources[0] == 'hardcodedmds':
        # args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower']
        n_cls = 712 + 70 + 140 + 33 + 994 + 883 + 241 + 71
        n_cls = 3144
    else:
        raise NotImplementedError
    print(f'{n_cls=}')
    assert n_cls != 0
    return n_cls
