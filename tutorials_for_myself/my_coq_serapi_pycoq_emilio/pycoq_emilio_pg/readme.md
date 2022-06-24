# 

```bash
opam install --deps-only coq-serapi
opam install -y pythonlib
```

```bash
make install && dune build examples/test.py && dune exec -- python3 _build/default/examples/test.py
```