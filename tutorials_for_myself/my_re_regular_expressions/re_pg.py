"""
very nive interface to try regexes: https://regex101.com/
"""
# %%
"""Simple if statement with a regex"""
import re

regex = r"\s*Proof.\s*"
contents = ['Proof.\n', '\nProof.\n']
for content in contents:
    assert re.match(regex, content), f'Failed on {content=} with {regex=}'
    if re.match(regex, content):
        print(content)

# %%
import re

"""
\w
For Unicode (str) patterns:
Matches Unicode word characters; this includes most characters that can be part of a word in any language, as well as numbers and the underscore. 
If the ASCII flag is used, only [a-zA-Z0-9_] is matched.
"""

# unicode_letter = f'[^\W\d_]'  # https://stackoverflow.com/questions/8923949/matching-only-a-unicode-letter-in-python-re
unicode_letter = f'\w'  # https://stackoverflow.com/questions/8923949/matching-only-a-unicode-letter-in-python-re
first_letter = f'[a-zA-Z]|_|{unicode_letter}'
subsequent_letter = f'{first_letter}|\d|\'|{unicode_letter}'
ident = f'{first_letter}({subsequent_letter})*'

# regex = f'(\s*)(Theorem|Lemma)(\s*)(\w+)(\s*:)'
regex = f'(\s*)(Theorem|Lemma)(\s*)({ident})(\s*:)'
print(f'{regex=}')
# regex = '(\s*)(Theorem|Lemma)(\s*)(.+)(\s*:)'
contents = [
    '\n\nTheorem double:',
    '\n\nTheorem double\': ',
    '\n\nTheorem double: forall x: bool, negb (negb x) = x.',
    '\n\nTheorem double\': forall x: bool, negb (negb x) = x.',
    'Theorem double:',
]
for content in contents:
    print('----')
    print(f'{content=}')
    assert re.match(regex, content), f'Failed on {content=} with {regex=}'
    m = re.match(regex, content)
    ret = m.groups()
    print(f'{ret=}')
    print(f'{len(ret)=}')
    print(ret[3])

# %%
# tokenizer

import re

terms: list[str] = [
    "(___hole 0 ?GOAL)",
    "(___hole 0 (fun n : nat => ___hole 1 (___hole 2 eq_refl : 0 + n = n)))",
    """
    (___hole 0
       (fun n : nat =>
        ___hole 1
          (nat_ind (fun n0 : nat => n0 + 0 = n0)
             (___hole 2 (___hole 3 eq_refl : 0 + 0 = 0))
             (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
             (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
              ___hole 2
                (___hole 5
                   (eq_ind_r (fun n1 : nat => S n1 = S n0) 
                      (___hole 6 eq_refl) IHn)
                 :
                 S n0 + 0 = S n0)) n)))
    """
]

# regex: str = "((\.*)(___hole \d)(\.*))+"
nat: str = r'\d+'
hole: str = rf'\(?___hole {nat}?'
any_str_not_hole: str = rf"^{hole}.*|.*^{hole}|"
rterm: str = rf"({any_str_not_hole})* ({hole})* ({any_str_not_hole})*"
print(f'{rterm=}')
print(f'{rterm}')

for term in terms:
    print('----')
    print(f'{term=}')
    assert re.match(rterm, term), f'Failed on {term=} with {rterm=}'
    m = re.match(rterm, term)
    ret = m.groups()
    print(f'{ret=}')
    print(f'{len(ret)=}')
    break
# %%

from nltk.tokenize import SExprTokenizer

seq_tokens: list[str] = SExprTokenizer().tokenize('(a b (c d)) e f (g)')
print(seq_tokens)
