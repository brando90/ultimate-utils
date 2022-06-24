(* print_string "Hello world!\n";; *)

(*
Running a file without compilation
https://discuss.ocaml.org/t/how-to-run-ml-ocaml-file-without-compiling/4311
*)
open Format;;

(* print_string "Hello world!\n";; *)

(* printf "Hello\n";; *)
(* printf "int value %d %s\n" 2 "string";; *)

type int_list =
| IntNil
| IntCons of int * int_list;;

IntNil;;
IntCons(1, IntNil);;
let lst1 = IntCons(2, IntNil);;
IntCons (1, (IntCons (2, IntNil)));;

let rec length l =
match l with
| IntNil -> 0
| IntCons (h, nl) -> 1 + length nl;;

let rec length2 (l:int_list) =
match l with
| IntNil -> 0
| IntCons (h, nl) -> 1 + length nl;;

(* Polymorphism / parametric polymorphism

'a is a type variable
*)

type 'a alist =
| ANil
| ACons  of 'a * ('a alist);;

type 't tlist =
| TNil
| TCons  of 't * ('t tlist);;

TNil;;
TCons(1, TNil);;

let id (x:'a) = x;;

(*
utop # let id (x:'a) = x;;
val id : 'a -> 'a = <fun>
*))