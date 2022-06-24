(* resource: http://xahlee.info/ocaml/types.html *)

(* you can put types in functions *)
let f (n:int) = n + 1;;

(*
TYPES
You can define your own type, by assigning a type expression to a name.
The syntax is this:
type name = type_expression.
*)

(*
Type constructor:
The simplest type is just a arbitrary letter sequence,
which is called “type constructor”.
A type constructor's first letter must be Capitalized.
*)

type t1 = X;; (* this defines the type t1 and only values of this type is the Symbol "X"*)
type t2 = Alice;; (* t2 is also a type and only value is Alice *)

(*
Type expression can use the operator |, which means “alternative”, “one of”, or “or”.
*)

type my_brothers = Andres | Alonso;;
type my_dogs = Max | Obamita;;

type foo =
  | Nothing (* Constnat *)
  | Int of int (* Int constructor with value Int int *)
  | Pair of int * int (* Pair constructor with value as pair of ints *)
  | String of string;; (* constructor String that has the value string *)

let nada = Nothing;;
(*
# let nada = Nothing;; declares a variable nada of type foo with value Nothing
val nada : foo = Nothing
*)
let x = Int 1;;
(*
# let x = Int 1;; declares variable x of type foo with value Int 1
val x : foo = Int 1
*)
let pair12 = Pair (1, 2);;
let hello = String "Hello!!!";;

(* Recursive variants *)

type binary_tree =
  | Leaf of int
  | Tree of binary_tree * binary_tree;;

Leaf 1;;
Tree ( Leaf 1, Leaf 2);;
Tree ( Tree(Leaf 1, Leaf 2), Leaf 3);;

let rec first_leaf_value tree =
match tree with
| Leaf n -> n
| Tree(left_tree, right_tree) -> first_leaf_value left_tree;;

first_leaf_value (Tree ( Tree(Leaf 1, Leaf 2), Leaf 3));;

type 'a list =
  | Nil
  | Cons of 'a * ('a list);;

Nil;;
Cons (1, Nil);; (* [1] 1::Nil *)
Cons (1, Cons(2, Nil));;

(* -- Dummy SerAPI test ground -- *)

(* type print_opt with 2 values*)
type print_opt =
| PrettyPrint
| MachinePrint;;

(* type coq obj with CoqStr as constructor to create values of the formal form CoqStr any string etc.*)
type coq_obj =
| CoqStr of string
| CoqAstSexp  of string;;

(* type dummy_serapi_cmd with 3 constructors that say how to create values for dumm_serapi_cmd (data) type *)
type dummy_serapi_cmd =
| Print of print_opt * coq_obj
| Add of string (* just dummy not meant to be real, obviously options etc are missing *)
| Exec of string (* just dummy not meant to be real, obviously options etc are missing *)
;;


(* make a command example! *)
let test_cmd = Print(PrettyPrint, (CoqStr "Lemma addn0 n : n + 0 = n."));;
test_cmd;;
