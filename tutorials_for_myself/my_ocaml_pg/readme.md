# Ocaml pg

see: https://github.com/brando90/cs421/tree/master/playground

## Installing Utop & Ocaml

```bash
brew install opam
# conda install -c conda-forge opam
# apt-get update apt-get install -y --no-install-recommends opam 
opam init
# opam init --disable-sandboxing
# opam switch create my_ocaml_switch
eval $(opam env)
# or eval `opam env`

opam install -y dune utop ocaml-lsp-server
opam install -y utop
```
ref:
- https://opam.ocaml.org/blog/about-utop/
- https://ocaml.org/docs/up-and-running

## Running OCaml without compilation

Go to the my_ocaml_pg:
```
cd ~/ultimate-utils/tutorials_for_myself/my_ocaml_pg
```

then

```
ocaml hello_world.ml
```
or 
```
utop hello_world.ml
```

If you have issues for printing see the printing in OCaml section.

Ref: https://discuss.ocaml.org/t/how-to-run-ml-ocaml-file-without-compiling/4311

## Printing in OCaml

Seems that due to losing typing during compilation (not really sure of the details) printing
with `ocaml scripy.ml` is not easy.

For now it seems the easiest to play around is write in your script and then copy paste into utop.

Another option is to open utop and "import" (e.g `#use "ocaml_pg.ml";;` in utop) your script every time. 
The disadvantage is that perhaps some of your code will be remembered in your utop session.

Ref:
- https://discuss.ocaml.org/t/how-does-one-print-any-type/4362/11

Maybe later dum...https://opam.ocaml.org/packages/dum/ 

## My CS421 stuff

https://github.com/brando90/cs421 

## Use vs Open vs Require

ref:
    - https://stackoverflow.com/questions/42631912/whats-the-difference-between-include-require-and-open-in-ocaml
    - https://discuss.ocaml.org/t/how-to-run-ml-ocaml-file-without-compiling/4311/18?u=brando90
