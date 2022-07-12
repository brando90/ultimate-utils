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

# check what happens if I send terms without definition of __hole

rlwrap sertop --printer=human

(Add () "Definition __hole {A:Type} (n:nat) (v:A) := v.")
(Exec 2)


(Parse () "(__hole 0 ?Goal).")

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

(Parse ((ontop 1)) "Definition id := fun x : nat => x.")
...
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

order to get the body of the vernac def and ast:
1. get VernacDefinition from expr (https://coq.github.io/doc/v8.11/api/coq/Vernacexpr/index.html#type-vernac_control -> https://coq.github.io/doc/v8.11/api/coq/Vernacexpr/index.html#type-vernac_control)
2. inside VernacDefinition go to DefineBody cons (3rd index so x[2]) https://coq.github.io/doc/v8.13/api/coq/Vernacexpr/index.html#type-vernac_expr
3. inside DefineBody go to the 3rd argument again (Constrexpr.constr_expr -- this one is where all the Coq Cons are! https://coq.github.io/doc/v8.13/api/coq/Constrexpr/index.html#type-constr_expr_r)
4. This gives the Coq Ast I want to refine. Let's move next to see how it looks like with an actual refine. 

# try the string -> proof term using parse with refined

rlwrap sertop --printer=human

(Add () "Definition __hole {A:Type} (n:nat) (v:A) := v.")
(Exec 2)

(Parse ((ontop 2)) "(__hole 0 ?Goal).")

rlwrap sertop --printer=human

(Add () "
Definition __hole {A:Type} (n:nat) (v:A) := v.
Theorem add_easy_0:
forall n:nat,
  0 + n = n.
  refine (__hole 0 _).
  Show Proof.")
(Exec 5)

(Parse ((ontop 5)) "(__hole 0 ?Goal).")

(Answer 2 Ack)
(Answer 2
 (ObjList
  ((CoqAst
    ((v
      ((control ()) (attrs ())
       (expr
        (VernacExtend (VernacSolve 0)
         ((GenArg raw (OptArg (ExtraArg ltac_selector)) ())
          (GenArg raw (OptArg (ExtraArg ltac_info)) ())
          (GenArg raw (ExtraArg tactic)
           (TacArg
            ((v
              (TacCall
               ((v
                 (((v (Ser_Qualid (DirPath ()) (Id __hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 7)))))
                  ((ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 8) (ep 9)))))))
                   (ConstrMayEval
                    (ConstrTerm
                     ((v (CEvar (Id Goal) ()))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 10) (ep 15))))))))) )
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 15)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 15)))))))
          (GenArg raw (ExtraArg ltac_use_default) false))))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 17)))))))))
(Answer 2 Completed)

(Print ((sid 4) (pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))

(Print () (
(((v (Ser_Qualid (DirPath ()) (Id __hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 7)))))
                  ((ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 8) (ep 9)))))))
                   (ConstrMayEval
                    (ConstrTerm
                     ((v (CEvar (Id Goal) ()))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 10) (ep 15)))))))))
)
)

# dynamic parser

(___hole 0 ?Goal)
(___hole 0 (fun n : nat => ___hole 1 ?Goal))
(___hole 0 (fun n : nat => ___hole 1 (___hole 2 ?Goal : 0 + n = n)))


(___hole 0 (fun n : nat =>  ___hole 1 (___hole 2 eq_refl : 0 + n = n) ))



(___hole 0 ?Goal)
(___hole 0 (fun n : nat => ___hole 1 ?Goal))
(___hole 0
   (fun n : nat =>
	___hole 1
      (nat_ind (fun n0 : nat => n0 + 0 = n0)
         (___hole 2 (___hole 2 ?Goal0 : 0 + 0 = 0))
         (fun (n0 : nat) (IHn : n0 + 0 = n0) => ___hole 2 ?Goal@{n:=n0}) n)))
(___hole 0
   (fun n : nat =>
	___hole 1
      (nat_ind (fun n0 : nat => n0 + 0 = n0)
         (___hole 2 (___hole 2 eq_refl : 0 + 0 = 0))
         (fun (n0 : nat) (IHn : n0 + 0 = n0) => ___hole 2 ?Goal@{n:=n0}) n)))
(___hole 0
   (fun n : nat =>
	___hole 1
      (nat_ind (fun n0 : nat => n0 + 0 = n0)
         (___hole 2 (___hole 2 eq_refl : 0 + 0 = 0))
         (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
          ___hole 2 (___hole 4 ?Goal@{n:=n0} : S n0 + 0 = S n0)) n)))
(___hole 0
   (fun n : nat =>
	___hole 1
      (nat_ind (fun n0 : nat => n0 + 0 = n0)
         (___hole 2 (___hole 2 eq_refl : 0 + 0 = 0))
         (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
          ___hole 2
            (___hole 4
               (eq_ind_r (fun n1 : nat => S n1 = S n0)
                  (___hole 5 ?Goal@{n:=n0}) IHn)
             :
             S n0 + 0 = S n0)) n)))

(___hole 0
   (fun n : nat =>
	___hole 1
      (nat_ind (fun n0 : nat => n0 + 0 = n0)
         (___hole 2 (___hole 3 eq_refl : 0 + 0 = 0))
         (fun (n0 : nat) (IHn : n0 + 0 = n0) =>
          ___hole 2
            (___hole 5
               (eq_ind_r (fun n1 : nat => S n1 = S n0) 
                  (___hole 6 eq_refl) IHn)
             :
             S n0 + 0 = S n0)) n)))


# -- question to emilio


rlwrap sertop --printer=human

(Add () "
Theorem add_easy_0: forall n:nat, 0 + n = n. simpl. reflexivity. Show Proof.")
(Exec 5)

(Parse ((ontop 5)) "(fun n : nat => eq_refl).")
(Parse () "(fun n : nat => eq_refl).")
(Parse ((ontop 5)) "(fun n : nat => eq_refl)")

(Parse () "Definition id := (fun n : nat => eq_refl).")

# --

rlwrap sertop --printer=human

(Add () "
Theorem add_easy_0: forall n:nat, 0 + n = n. simpl. reflexivity. Show Proof.")
(Exec 5)

(Parse () "Definition id := (fun n : nat => eq_refl).")

(Answer 2 Ack)
(Answer 2
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
               (((v (Name (Id n)))
                 (loc
                  (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                    (line_nb_last 1) (bol_pos_last 0) (bp 22) (ep 23))))))
               (Default Explicit)
               ((v
                 (CRef
                  ((v (Ser_Qualid (DirPath ()) (Id nat)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 26) (ep 29)))))
                  ()))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 26) (ep 29)))))))
             ((v
               (CRef
                ((v (Ser_Qualid (DirPath ()) (Id eq_refl)))
                 (loc
                  (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                    (line_nb_last 1) (bol_pos_last 0) (bp 33) (ep 40)))))
                ()))
              (loc
               (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                 (line_nb_last 1) (bol_pos_last 0) (bp 33) (ep 40)))))))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
              (bol_pos_last 0) (bp 18) (ep 40)))))
          ())))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 42)))))))))
(Answer 2 Completed)