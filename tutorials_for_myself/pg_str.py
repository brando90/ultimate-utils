# %%

l: list[str] = ['opam', 'exec', '--switch', 'ocaml-variants.4.07.1+flambda_coq-serapi.8.11.0+0.11.1', '--', 'which',
                'coqc']

s: str = ' '.join(l)

print(s)
