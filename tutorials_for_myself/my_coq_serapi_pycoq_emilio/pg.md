# Playground for Coq-SerAPI

For every command, SerAPI will always reply with`(Answer tag Ack)`.

For a particular point on the document, you can query Coq for information about
it. Common query use cases are for example lists of tactics, AST, completion, etc...
Querying is done using the `[(Query (opts) query)]` command.

A Coq document is basically a list of sentences which are uniquely identified by a Stateid.t object; 
for our purposes this identifier is an integer.
Note that the parent is important for parsing as it may modify the parsing itself, 
for example it may be a Notation command.


## Emilio's Serapi
```
rlwrap sertop --printer=human
```

Prints current goals & LC all together:
```
(Add () "Lemma addn0 n : n + 0 = n.")
(Add () "Proof.")
(Exec 3)

# (Query (opts) query)
(Query ((sid 3) (pp ((pp_format PpStr)))) Goals)
(Query ((pp ((pp_format PpStr)))) Goals)
(Query () Goals)
```

Print CoqString from VP's s-exp object using SerAPI print:
```
(Add () "Lemma addn0 n : n + 0 = n.")
(Add () "Proof.")
(Exec 3)

(Query () Goals)
(Print  ((pp ((pp_format PpStr)))) )

(pp_ex (Print ((sid 4) (pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0))))))
(Print ((sid 4) (pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))
(Print ((pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))
# FAILS (Print (pp ((pp_format PpStr))) (CoqConstr (App (Rel 0) ((Rel 0)))))

(Print ((pp ((pp_format PpStr)))) (CoqObj))
# (Print ((pp ((pp_format PpStr)))) ( ObjList((CoqGoal((goals(((info((evar(Ser_Evar 3))(name())))(ty(Prod((binder_name(Name(Id x)))(binder_relevance Relevant))(Ind(((MutInd(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id bool))0)(Instance())))(App(Ind(((MutInd(MPfile(DirPath((Id Logic)(Id Init)(Id Coq))))(Id eq))0)(Instance())))((Ind(((MutInd(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id bool))0)(Instance())))(App(Const((Constant(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id negb))(Instance())))((App(Const((Constant(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id negb))(Instance())))((Rel 1)))))(Rel 1)))))(hyp()))))(stack())(shelf())(given_up())(bullet())))) ))
(Print ((pp ((pp_format PpStr)))) (CoqGoal((goals(((info((evar(Ser_Evar 3))(name())))(ty(Prod((binder_name(Name(Id x)))(binder_relevance Relevant))(Ind(((MutInd(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id bool))0)(Instance())))(App(Ind(((MutInd(MPfile(DirPath((Id Logic)(Id Init)(Id Coq))))(Id eq))0)(Instance())))((Ind(((MutInd(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id bool))0)(Instance())))(App(Const((Constant(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id negb))(Instance())))((App(Const((Constant(MPfile(DirPath((Id TwoGoals)(Id LF))))(Id negb))(Instance())))((Rel 1)))))(Rel 1)))))(hyp()))))(stack())(shelf())(given_up())(bullet()))))
```

###

What about this algorithm 1 (from VPs to many round tripts to back serapi):

- go through all the parsed terms VP gave us.
- for each term, construct the right serapi-exp (replaced `[]` with `()`)
- test it with the `(Print (pp (pp_format PpStr)) Obj )`. There you go, you have a Coq pretty printed string object.

Note: if you query all goals at once you get one big string.
Is this true with PyCoq-SerApi?

TODO: can this be done with PyCoq-SerApi?

### PPT

goal: get ppt with query
```
rlwrap sertop --printer=human

(Add () "Lemma addn0 n : n + 0 = n.")
(Exec 2)
(Query ((sid 2) (pp ((pp_format PpStr)))) Goals)

(Add () "Lemma addn0 n : 0 + n = n. Proof. intros. simpl. reflexivity.")
(Exec 6)
(Query ((pp ((pp_format PpStr)))) Goals)

https://github.com/ejgallego/coq-serapi/issues/270
(Add () "Lemma addn0 n : 0 + n = n. Proof. intros. simpl.")
(Exec 5)
(Query ((pp ((pp_format PpStr)))) Goals)
(Query ((pp ((pp_format PpStr)))) Proof)


(Query ((sid 1) (pp ((pp_format PpStr)))) Goals)
(Query ((sid 2) (pp ((pp_format PpStr)))) Goals)
(Query ((sid 3) (pp ((pp_format PpStr)))) Goals)
(Query ((pp ((pp_format PpStr)))) Goals)
(Query () Proof)
```

get ppt
```
rlwrap sertop --printer=human

(Add () "Lemma addn0 n : 0 + n = n. Proof. intros. simpl. Show Proof.")

RESPONSE:
(Exec 6)
(Answer 1 Ack)
(Feedback
 ((doc_id 0) (span_id 6) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 5) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 4) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 3) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 2) (route 0) (contents (ProcessingIn master))))
(Feedback ((doc_id 0) (span_id 1) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 2) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 3) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 4) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 5) (route 0) (contents Processed)))
(Feedback
 ((doc_id 0) (span_id 6) (route 0)
  (contents
   (Message (level Notice) (loc ())
    (pp
     (Pp_box (Pp_hovbox 1)
      (Pp_glue
       ((Pp_string "(")
        (Pp_box (Pp_hovbox 0)
         (Pp_glue
          ((Pp_box (Pp_hovbox 2)
            (Pp_glue
             ((Pp_tag constr.keyword (Pp_string fun)) (Pp_print_break 1 0)
              (Pp_box (Pp_hovbox 1)
               (Pp_glue
                ((Pp_string "n : ") (Pp_tag constr.variable (Pp_string nat))))))))
           (Pp_print_break 1 0) (Pp_string =>) (Pp_print_break 1 0)
           (Pp_box (Pp_hovbox 0)
            (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
        (Pp_string ")")))))
    (str "(fun n : nat => ?Goal)")))))
(Feedback ((doc_id 0) (span_id 6) (route 0) (contents Processed)))
(Answer 1 Completed)

rlwrap sertop --printer=human
Inductive bool: Type :=\n| true\n| false.\n
\n\nDefinition negb (x: bool): bool :=\n  match x with\n  | true => false\n  | false => true\n  end.\n
\n\nTheorem double: forall x: bool, negb (negb x) = x.\n
Proof.\n

rlwrap sertop --printer=human

(Add () "Lemma addn0 n : 0 + n = n. Proof.")
(Exec 3)
(Add () "Show Proof.")
(Exec 4)


rlwrap sertop --printer=human

(Add () "Lemma addn0: forall n: nat, 0 + n = n. Proof.")
(Exec 3)
(Add () "Show Proof.")
(Exec 4)

```

```
(Add () "From Coq Require Import ssreflect ssrfun ssrbool.
Theorem comm: 
    forall (n:nat) (m:nat),
    n + m = m + n.
Proof. 
    intros.
    have H0: True by auto.
    have H1: 1 = 1 by auto." )
(Exec 7)
(Query ((pp ((pp_format PpStr)))) Goals)

(Add () "From Coq Require Import ssreflect ssrfun ssrbool.
Theorem comm: 
    forall (n:nat) (m:nat),
    n + m = m + n.
Proof. 
    intros.
    have H0: True by auto.
    have H1: 1 = 1 by auto.
    have H2: 12 = 12." )
(Exec 8)
(Query ((pp ((pp_format PpStr)))) Goals)
```

```
(Add () "
Definition __hole {A:Type} (n:nat) (v:A) := v.
Theorem add_easy_0'':
forall n:nat,
  0 + n = n.
Proof.
    refine (__hole 0 _).
    intros;
    refine (__hole 1 _).
    Show Proof.
")
(Exec 7)

(Feedback
 ((doc_id 0) (span_id 7) (route 0)
  (contents
   (Message (level Notice) (loc ())
    (pp
     (Pp_box (Pp_hovbox 1)
      (Pp_glue
       ((Pp_string "(")
        (Pp_box (Pp_hovbox 2)
         (Pp_glue
          ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
           (Pp_string 0) (Pp_print_break 1 0)
           (Pp_box (Pp_hovbox 1)
            (Pp_glue
             ((Pp_string "(")
              (Pp_box (Pp_hovbox 0)
               (Pp_glue
                ((Pp_box (Pp_hovbox 2)
                  (Pp_glue
                   ((Pp_tag constr.keyword (Pp_string fun))
                    (Pp_print_break 1 0)
                    (Pp_box (Pp_hovbox 1)
                     (Pp_glue
                      ((Pp_string "n : ")
                       (Pp_tag constr.variable (Pp_string nat))))))))
                 (Pp_print_break 1 0) (Pp_string =>) (Pp_print_break 1 0)
                 (Pp_box (Pp_hovbox 2)
                  (Pp_glue
                   ((Pp_tag constr.variable (Pp_string __hole))
                    (Pp_print_break 1 0) (Pp_string 1) (Pp_print_break 1 0)
                    (Pp_box (Pp_hovbox 0)
                     (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal)))))))))))
              (Pp_string ")")))))))
        (Pp_string ")")))))
    (str "(__hole 0 (fun n : nat => __hole 1 ?Goal))")))))
(Feedback ((doc_id 0) (span_id 7) (route 0) (contents Processed)))
(Answer 1 Completed)
```

```
(Add () "
Definition __hole {A:Type} (n:nat) (v:A) := v.
\n\n\nTheorem refined___double: forall x: bool, negb (negb x) = x.\n refine (__hole 0 _).\n  destruct x;\n refine (__hole 2 _).\n  - simpl;  refine (__hole 3 _).\nreflexivity;\n refine (__hole 4 _).\n  - simpl;  refine (__hole 5 _).\nreflexivity;\n refine (__hole 6 _).\nShow Proof.
")
(Exec 12)
```

```
rlwrap sertop --printer=human

(Add () "
Definition __hole {A:Type} (n:nat) (v:A) := v.
Theorem add_easy_0'':
forall n:nat,
  0 + n = n.
Proof.
    refine (__hole 0 _).
    Show Proof.
")

(Exec 6)

(Answer 1 Ack)
(Feedback
 ((doc_id 0) (span_id 6) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 5) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 4) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 3) (route 0) (contents (ProcessingIn master))))
(Feedback
 ((doc_id 0) (span_id 2) (route 0) (contents (ProcessingIn master))))
(Feedback ((doc_id 0) (span_id 1) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 2) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 3) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 4) (route 0) (contents Processed)))
(Feedback ((doc_id 0) (span_id 5) (route 0) (contents Processed)))
(Feedback
 ((doc_id 0) (span_id 6) (route 0)
  (contents
   (Message (level Notice) (loc ())
    (pp
     (Pp_box (Pp_hovbox 1)
      (Pp_glue
       ((Pp_string "(")
        (Pp_box (Pp_hovbox 2)
         (Pp_glue
          ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
           (Pp_string 0) (Pp_print_break 1 0)
           (Pp_box (Pp_hovbox 0)
            (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
        (Pp_string ")")))))
    (str "(__hole 0 ?Goal)")))))
(Feedback ((doc_id 0) (span_id 6) (route 0) (contents Processed)))
(Answer 1 Completed)


(Print ((pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))
```

(pp (Pp_string ")"))

(pp
 (Pp_box (Pp_hovbox 1)
  (Pp_glue
   ((Pp_string "(")
    (Pp_box (Pp_hovbox 2)
     (Pp_glue
      ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
       (Pp_string 0) (Pp_print_break 1 0)
       (Pp_box (Pp_hovbox 0)
        (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
    (Pp_string ")")))))


(Print ((pp ((pp_format PpStr))))     (CoqPp (
(pp
 (Pp_box (Pp_hovbox 1)
  (Pp_glue
   ((Pp_string "(")
    (Pp_box (Pp_hovbox 2)
     (Pp_glue
      ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
       (Pp_string 0) (Pp_print_break 1 0)
       (Pp_box (Pp_hovbox 0)
        (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
    (Pp_string ")")))))
) )         )


(Print ((pp ((pp_format PpStr))))     (CoqPp ((pp (Pp_box (Pp_hovbox 1) (Pp_glue ((Pp_string "(") (Pp_box (Pp_hovbox 2) (Pp_glue ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0) (Pp_string 0) (Pp_print_break 1 0) (Pp_box (Pp_hovbox 0) (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal)))))))) (Pp_string ")"))))) ) )         )

(Print ((pp (pp_format PpStr))) (CoqPp (pp (Pp_string "hi"))) )

# found the pp cons I think: https://github.com/ejgallego/coq-serapi/blob/b222c1f821694175273d3a295a04abf9454f6727/serlib/ser_pp.ml
# just construct some CoqPp that prints back
(Print ((pp (pp_format PpStr))) (CoqPp ()) )

(Feedback
 ((doc_id 0) (span_id 6) (route 0)
  (contents
   (Message (level Notice) (loc ())
    (pp
     (Pp_box (Pp_hovbox 1)
      (Pp_glue
       ((Pp_string "(")
        (Pp_box (Pp_hovbox 2)
         (Pp_glue
          ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
           (Pp_string 0) (Pp_print_break 1 0)
           (Pp_box (Pp_hovbox 0)
            (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
        (Pp_string ")")))))
    (str "(__hole 0 ?Goal)") ))))

# only ?GOAL is simpler

(Add () "
Theorem add_easy_0'':
forall n:nat,
  0 + n = n.
Proof.
    Show Proof.
")

(Exec 4)

(Feedback
 ((doc_id 0) (span_id 4) (route 0)
  (contents
   (Message (level Notice) (loc ())
    (pp
     (Pp_box (Pp_hovbox 0)
      (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))
    (str ?Goal)))))

(pp
 (Pp_box (Pp_hovbox 0)
  (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))

# ?GOAL 1 step forward

(Add () "
Definition __hole {A:Type} (n:nat) (v:A) := v.
Theorem add_easy_0'':
forall n:nat,
  0 + n = n.
Proof.
    refine (__hole 0 _).
    Show Proof.
")

(Exec 6)

(Feedback
 ((doc_id 0) (span_id 6) (route 0)
  (contents
   (Message (level Notice) (loc ())
    (pp
     (Pp_box (Pp_hovbox 1)
      (Pp_glue
       ((Pp_string "(")
        (Pp_box (Pp_hovbox 2)
         (Pp_glue
          ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
           (Pp_string 0) (Pp_print_break 1 0)
           (Pp_box (Pp_hovbox 0)
            (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
        (Pp_string ")")))))
    (str "(__hole 0 ?Goal)")))))

(pp
 (Pp_box (Pp_hovbox 1)
  (Pp_glue
   ((Pp_string "(")
    (Pp_box (Pp_hovbox 2)
     (Pp_glue
      ((Pp_tag constr.variable (Pp_string __hole)) (Pp_print_break 1 0)
       (Pp_string 0) (Pp_print_break 1 0)
       (Pp_box (Pp_hovbox 0)
        (Pp_tag constr.evar (Pp_glue ((Pp_string ?Goal))))))))
    (Pp_string ")")))))

# abadon that attempt, perhaps lets recall what serapi returns from the query proof object?

(Add () "
Theorem add_easy_0'':
forall n:nat,
  0 + n = n.
Proof.
    Show Proof.
")

(Exec 4)

(Query ((pp ((pp_format PpStr)))) Proof)

(Print ((sid 4) (pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))

# -- idea: give it the proof term as a string from the feedback response to print but have Print construct an parsable ast for us
(Print ((pp ((pp_format PpStr)))) (CoqString  (str ?Goal))))
(Print ((pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))

(Print ((pp ((pp_format PpStr)))) (CoqString ?Goal))

(Print ((pp ((pp_format PpStr)))) (CoqString "fun x: nat => x."))

(Print ((pp ((pp_format PpSer)))) (CoqString "fun x: nat => x."))

(Print ((pp ((pp_format PpCoq)))) (CoqString "fun x: nat => x."))

```
seems that the print option won't give me a traversable ast -- even if I get the proof term mysel (from the feedback) and send it back to serapi. 
```

# -- What does (Parse ...) do?
(Parse parse_opt str)


(Parse () "fun x: nat => x.")
(Parse ((ontop 1)) "fun x : nat => x.")

(Parse ((ontop 1)) "Definition id := fun x : nat => x.")

 (ObjList
  ((CoqAst
    ((v
      ((control ()) (attrs ())
       (expr
        (VernacDefinition (NoDischarge Definition)
         (((v (Name (Id id)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
              (bol_pos_last 0) (bp 11) (ep 13)))))
          ())
         (DefineBody () ()
          ((v
            (CLambdaN
             ((CLocalAssum
               (((v (Name (Id x)))
                 (loc
                  (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                    (line_nb_last 1) (bol_pos_last 0) (bp 21) (ep 22))))))
               (Default Explicit)
               ((v
                 (CRef
                  ((v (Ser_Qualid (DirPath ()) (Id nat)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 25) (ep 28)))))
                  ()))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 25) (ep 28)))))))
             ((v
               (CRef
                ((v (Ser_Qualid (DirPath ()) (Id x)))
                 (loc
                  (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                    (line_nb_last 1) (bol_pos_last 0) (bp 32) (ep 33)))))
                ()))
              (loc
               (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                 (line_nb_last 1) (bol_pos_last 0) (bp 32) (ep 33)))))))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
              (bol_pos_last 0) (bp 17) (ep 33)))))
          ())))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 34))))))))

(CoqAst
    ((v
      ((control ()) (attrs ())
       (expr
        (VernacDefinition (NoDischarge Definition)
         (((v (Name (Id id)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
              (bol_pos_last 0) (bp 11) (ep 13)))))
          ())
         (DefineBody () ()
          ((v
            (CLambdaN
             ((CLocalAssum
               (((v (Name (Id x)))
                 (loc
                  (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                    (line_nb_last 1) (bol_pos_last 0) (bp 21) (ep 22))))))
               (Default Explicit)
               ((v
                 (CRef
                  ((v (Ser_Qualid (DirPath ()) (Id nat)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 25) (ep 28)))))
                  ()))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 25) (ep 28)))))))
             ((v
               (CRef
                ((v (Ser_Qualid (DirPath ()) (Id x)))
                 (loc
                  (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                    (line_nb_last 1) (bol_pos_last 0) (bp 32) (ep 33)))))
                ()))
              (loc
               (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                 (line_nb_last 1) (bol_pos_last 0) (bp 32) (ep 33)))))))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
              (bol_pos_last 0) (bp 17) (ep 33)))))
          ())))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 34))))))
