
# %%
from lark import Lark, Tree

parser: Lark = Lark(r"""
    hole_0: "(" "___hole 0" anything hole_1 anything ")"
    hole_1: "(" "___hole 1" anything hole_2 anything ")" | "___hole 1" anything hole_2 anything
    hole_2: "(" "___hole 2" anything end anything ")" | "___hole 2" anything end anything
    end: anything 
    
    anything: /.+/
    // anything: "(fun n : nat =>" | "eq_refl : 0 + n = n"
    
    %import common.ESCAPED_STRING 
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """,
    start='hole_0',
    lexer="dynamic_complete"
    )

test_strings: list[str] = [
    "(___hole 0 (fun n : nat => ___hole 1 (___hole 2 eq_refl : 0 + n = n)))"
]

for test_string in test_strings:
    print(f'{test_string=}')
    tree: Tree = parser.parse(test_string)
    print(tree.pretty())
