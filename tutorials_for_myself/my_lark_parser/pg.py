#%%
from lark import Lark, Tree

parser: Lark = Lark(r"""
    start: "a" b c
    
    b: "b"
    c: "c"

    anything: /.+?/ | ESCAPED_STRING

    %import common.ESCAPED_STRING 
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='start')

test_strings: list[str] = ["abc"]

for test_string in test_strings:
    print(f'{test_string=}')
    tree: Tree = parser.parse(test_string)
    print(tree.pretty())
