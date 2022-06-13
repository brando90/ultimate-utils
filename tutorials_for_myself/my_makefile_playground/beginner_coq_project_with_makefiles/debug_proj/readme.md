# Debug proj

It's a simple coq project just to play around with building coq projects using the recommended instructions from the
coq utitlites documentation https://coq.inria.fr/refman/practical-tools/utilities.html#the-grammar-of-coqproject.

## Install

To build this coq project first have coq build the `CoqMakefile` for you (that you will not usually touch):
```bash
coq_makefile -f _CoqProject -o CoqMakefile
```
this automatically generates a make file make file named `CoqMakefile` that you will not usually edit.
The make file I am managing is the `Makefile` file.

Now we can build our project with:
```bash
make
```
and to clean/remove the build do
```bash
make clean
```
which I assume is somehow managed/run properly because we are running `CoqMakefile` through out makefile.
See it's contents. 

##

TODO:
- how do we include coq (external) dependencies? my guess is that other's would deal with that in
  their manual `Makefile` (not `CoqMakefile` which is created automagically with `coq_makefile -f _CoqProject -o CoqMakefile`).

## Useful refs:

- makefiles tutorial: https://makefiletutorial.com/#makefile-cookbook
- https://coq.inria.fr/refman/practical-tools/utilities.html#building-a-coq-project