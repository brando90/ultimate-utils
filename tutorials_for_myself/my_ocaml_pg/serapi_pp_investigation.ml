(*
Figure out how PP strings for serapi are constructed.
*)

type 'a list =
  | Nil
  | Cons of 'a * ('a list);;

Nil;;
Cons (1, Nil);;  (* [1] 1::Nil *)
Cons (1, Cons(2, Nil));;

open Format;;
type 'a pp = Format.formatter -> 'a -> unit;;

(*val pp_str      :                            string   pp;;*)
(*val pp_opt      :                'a pp -> ('a option) pp;;*)
(*
odd it doesnt let me do this:

─( 12:33:42 )─< command 11 >────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
utop # type 'a pp = Format.formatter -> 'a -> unit;;
type 'a pp = formatter -> 'a -> unit
─( 12:33:49 )─< command 12 >────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
utop # pp int;;
Error: Unbound value pp


*)