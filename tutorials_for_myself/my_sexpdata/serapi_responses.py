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
from sexpdata import loads, dumps

res: str = "( ObjList ( (CoqString\"none\\n============================\\nforall x : bool, negb (negb x) = x\") )  )"
print(res)
x = loads(res)
print(x)
print()
print(x[1][0][1])
print()

res: str = "(x y z)"
y = loads(res)
print(y)