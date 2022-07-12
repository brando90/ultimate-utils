(*
Note that when you define a type as in

type 'a my_type =
| Cons0
| Cons1 of 'a * int
| Cons2 of 'a * 'a my_type;;

You are allowing a type that takes in any type 'a. So for example, if you were to create a new type say

type my_new_type = int my_type

inspired from:

and vernac_control = vernac_control_r CAst.t ocaml syntax means.

then it means you are defining a new type using the old type that you know at static type!

TODO: confirm what things are figured out during runtime. e.g. int list.
*)