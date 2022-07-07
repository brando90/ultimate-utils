# %%

# https://stackoverflow.com/questions/72901395/how-to-match-any-terminal-string-with-a-regex-in-the-python-lark-parser-and-the

from lark import Lark, Tree

parser: Lark = Lark(r"""
    rterm: "(___hole 0" ANYTHING  ")"

    ANYTHING: /.+/

    %import common.ESCAPED_STRING 
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='rterm', lexer="dynamic_complete")

test_strings: list[str] = [
    "(___hole 0 (fun n : nat => ___hole 1 (___hole 2 eq_refl : 0 + n = n)))"
]

print('-- Start test')
for test_string in test_strings:
    print(f'{test_string=}')
    tree: Tree = parser.parse(test_string)
    print(tree.pretty())

print('--Done--')
