"""
In the usual parenthesized syntax of Lisp, an S-expression is classically defined as:
    1. an atom, or
    2. an expression of the form (x . y) where x and y are S-expressions.

My preferred one (avoid weird Pairs with one element and a nil)
    type sexp =
        | Atom of str
        | Pair of sexp * sexp;;

or the wikipedia one:
    type sexp =
        | Atom of atom
        | Pair of sexp * sexp;;
    type atom =
        | Nil
        | str;;

This definition reflects LISP's representation of a list as a series of "cells", each one an ordered pair.
In plain lists, y points to the next cell (if any), thus forming a list.
The recursive clause of the definition means that both this representation and the S-expression notation
can represent any binary tree.

(x y z)
stands for:
(x y z) = [x, y, z] = [ x, [y, z]]  [mine]
(x y z) = (x . (y . (z . NIL)))  [wikipedia]
** main take away is that a list can be represented as a binary tree where the left leafs start to expose the atoms/data.

e.g.
(+ 3 4)
stands for:

[*, [3, 4]] [mine]
(+ . (3 . (4 . NIL)) [wiki]

with fill constructors:
Pair(Atom(*), Pair(Atom(3), Atom(4))
Pair(+, Pair(4, Nil))


Note: that the sexpdata library doesn't actually work with binary trees literally. It seems that if you have a string
with parenthesis that looks like an array it gives it to you the intuitive way python would have it, as a multi element
array and not as a binary tree (at least not to the user).
"""
# %%

# from sexpdata import loads, dumps
#
# res: str = "( ObjList ( (CoqString\"none\\n============================\\nforall x : bool, negb (negb x) = x\") )  )"
# print(res)
# x = loads(res)
# print(x)
# print()
# print(x[1][0][1])
# print()
#
# res: str = "(x y z)"
# y = loads(res)
# print(y)

#%%
"""
type answer = 
| Feedback of feedback

type feedback = {
...
contents : feedback_content;
}

type feedback_content = 
...
| Message of {
level : Feedback.level;
loc : Loc.t option;
pp : Pp.t;
str : string;
}
"""
from pprint import pprint

def extract_proof_term(serapi_response: str) -> str:
    from sexpdata import loads
    res: str = serapi_response
    print(res)
    _res: list = loads(res)
    feedback: list = _res[-1]
    assert len(feedback) == 4
    print(f"{len(feedback)=}")
    pprint(feedback)
    contents: list = feedback[-1]
    pprint(contents)
    print(f'{len(contents)=}')
    assert len(contents) == 2
    message: list = contents[-1]
    assert len(message) == 5
    string: list = message[-1]
    assert string[0].value() == 'str'
    if isinstance(string[-1], str):
        ppt: str = string[-1]
    else:
        ppt: str = string[-1].value()
    print(ppt)
    return ppt

res0: str = """(Feedback((doc_id 0)(span_id 6)(route 0)(contents(Message(level Notice)(loc())(pp(Pp_box(Pp_hovbox 0)(Pp_tag constr.evar(Pp_glue((Pp_string ?Goal))))))(str ?Goal)))))\n"""
print(res0)
ppt0: str = extract_proof_term(res0)
print(ppt0)

res1: str = """(Feedback((doc_id 0)(span_id 8)(route 0)(contents(Message(level Notice)(loc())(pp(Pp_box(Pp_hovbox 1)(Pp_glue((Pp_string"(")(Pp_box(Pp_hovbox 0)(Pp_glue((Pp_box(Pp_hovbox 2)(Pp_glue((Pp_tag constr.keyword(Pp_string fun))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 1)(Pp_glue((Pp_string"x : ")(Pp_tag constr.variable(Pp_string bool))))))))(Pp_print_break 1 0)(Pp_string =>)(Pp_print_break 1 0)(Pp_box(Pp_vbox 0)(Pp_glue((Pp_box(Pp_hvbox 0)(Pp_glue((Pp_tag constr.keyword(Pp_string match))(Pp_print_break 1 2)(Pp_box(Pp_hovbox 0)(Pp_glue((Pp_box(Pp_hovbox 0)(Pp_glue((Pp_tag constr.variable(Pp_string x))(Pp_print_break 1 0)(Pp_tag constr.keyword(Pp_string as))(Pp_print_break 1 0)(Pp_string b))))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 2)(Pp_glue((Pp_tag constr.keyword(Pp_string return))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 1)(Pp_glue((Pp_string"(")(Pp_box(Pp_hovbox 0)(Pp_glue((Pp_box(Pp_hovbox 2)(Pp_glue((Pp_tag constr.variable(Pp_string negb))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 1)(Pp_glue((Pp_string"(")(Pp_box(Pp_hovbox 2)(Pp_glue((Pp_tag constr.variable(Pp_string negb))(Pp_print_break 1 0)(Pp_tag constr.variable(Pp_string b)))))(Pp_string")")))))))(Pp_tag constr.notation(Pp_string" ="))(Pp_print_break 1 0)(Pp_tag constr.variable(Pp_string b)))))(Pp_string")"))))))))))(Pp_print_break 1 0)(Pp_tag constr.keyword(Pp_string with)))))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 4)(Pp_glue((Pp_string"| ")(Pp_box(Pp_hovbox 0)(Pp_glue((Pp_box(Pp_hovbox 0)(Pp_tag constr.variable(Pp_string true)))(Pp_string" =>"))))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 0)(Pp_tag constr.evar(Pp_glue((Pp_string ?Goal))))))))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 4)(Pp_glue((Pp_string"| ")(Pp_box(Pp_hovbox 0)(Pp_glue((Pp_box(Pp_hovbox 0)(Pp_tag constr.variable(Pp_string false)))(Pp_string" =>"))))(Pp_print_break 1 0)(Pp_box(Pp_hovbox 0)(Pp_tag constr.evar(Pp_glue((Pp_string ?Goal0))))))))(Pp_print_break 1 0)(Pp_tag constr.keyword(Pp_string end))))))))(Pp_string")")))))(str"(fun x : bool =>\\n match x as b return (negb (negb b) = b) with\\n | true => ?Goal\\n | false => ?Goal0\\n end)")))))\n"""
print(res1)
ppt1: str = extract_proof_term(res1)
print(ppt1)

