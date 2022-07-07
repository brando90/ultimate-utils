# %%
from lark import Lark, Tree

json_parser: Lark = Lark(r"""
    value: dict
         | list
         | ESCAPED_STRING
         | SIGNED_NUMBER
         | "true" | "false" | "null"

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : ESCAPED_STRING ":" value

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='value')

text = '{"key": ["item0", "item1", 3.14]}'
tree: Tree = json_parser.parse(text)
print(tree.pretty())
