# Introduction

This package is for computing metrics, especially using adaptation of the original svcca, pwcca.
If you use anything in this file please cite the original authors:

```
@incollection{NIPS2017_7188,
title = {SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability},
author = {Raghu, Maithra and Gilmer, Justin and Yosinski, Jason and Sohl-Dickstein, Jascha},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {6076--6085},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability.pdf}
}

```

```
@incollection{NIPS2018_7815,
title = {Insights on representational similarity in neural networks with canonical correlation},
author = {Morcos, Ari and Raghu, Maithra and Bengio, Samy},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {5732--5741},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation.pdf}
}

```

# LICENSE

Due to original authors of svcca, pwcca, we are adding their license here and adding list of changes here instead of 
in the `./cca/NOTICE` file:

Changes:
    - all additions to the original svcca library are in file `uutils_cca_core_addendums.py`

## Licensing comment

My understanding from opensource SO:
> You can use (and modify) the codebase of the Apache component under the Apache license. You cannot re-license the 
> Apache code to use MIT in the future, because there are addnl restrictions in the Apache license (stating changes, 
> trademark use) which do not appear in the MIT license.
> I always recommend to have one file licenses.md (or .txt or whatever) either in the top folder or in a 
> 'licenses' subfolder, and include everything in that one file. In addition you should put a 'notices.md' into the 
> same folder with the copyright attribution, list of changes etc. Some people put the notices in the same file with 
> the licenses, but I think that makes it very difficult to read.

## LICENSE References
- https://opensource.stackexchange.com/q/12263/25395