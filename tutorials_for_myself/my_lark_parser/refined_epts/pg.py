
# %%
from lark import Lark, Tree

parser: Lark = Lark(r"""
    hole0: "(" "___hole 0" anything hole1 anything ")"
    hole1: "(" "___hole 1" anything hole2 anything ")" | "___hole 1" anything hole2 anything 
    hole2: "(" "___hole 2" anything end anything ")" | "___hole 2" anything end anything
    end: anything 
    
    anything: /.+/
    // anything: "(fun n : nat =>" | "eq_refl : 0 + n = n"
    
    %import common.ESCAPED_STRING 
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """,
    start='hole0',
    lexer="dynamic_complete"
    )

test_strings: list[str] = [
    "(___hole 0 (fun n : nat => ___hole 1 (___hole 2 eq_refl : 0 + n = n)))",
    # """
    # (___hole 0
    #    (fun n : nat =>
    #     ___hole 1
    #       (nat_ind (fun n0 : nat => n0 + 0 = n0)
    #          (___hole 2 (___hole 3 eq_refl : 0 + 0 = 0))
    #          (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
    #          (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
    #           ___hole 2
    #             (___hole 5
    #                (eq_ind_r (fun n1 : nat => S n1 = S n0)
    #                   (___hole 6 eq_refl) IHn)
    #              :
    #              S n0 + 0 = S n0)) n)))
    # """
]

for test_string in test_strings:
    print(f'{test_string=}')
    tree: Tree = parser.parse(test_string)
    print(tree.pretty())
