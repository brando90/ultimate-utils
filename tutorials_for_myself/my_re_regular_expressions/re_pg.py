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
