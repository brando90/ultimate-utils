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
