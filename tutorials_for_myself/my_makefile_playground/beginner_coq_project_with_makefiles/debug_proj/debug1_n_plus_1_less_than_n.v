(*Require Import Omega.

 Theorem t:
    forall n: nat, 1 + n > n.
 Proof.
 Show Proof.
 intro.
 Show Proof.
 omega.
 Show Proof.
 Qed.
 *)

Require Import Lia.

 Theorem t:
    forall n: nat, 1 + n > n.
 Proof.
 Show Proof.
 intro.
 Show Proof.
 lia.
 Show Proof.
 Qed.