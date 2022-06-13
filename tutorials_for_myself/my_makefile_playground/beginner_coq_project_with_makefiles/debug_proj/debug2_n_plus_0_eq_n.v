Theorem add_easy_induct_1:
forall n:nat,
  n + 0 = n.
Proof.
  Show Proof.
      intros.
  Show Proof.
  induction n as [| n' IH].
  Show Proof.
  - simpl.
    Show Proof.
    reflexivity.
    Show Proof.
  - simpl.
    Show Proof.
    rewrite -> IH.
    Show Proof.
    reflexivity.
    Show Proof.
Qed.