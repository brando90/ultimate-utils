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

# - dynamic parser
# https://github.com/FormalML/iit-term-synthesis/issues/6

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
          ())) )))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 42)))))))))
(Answer 2 Completed)

# --

goal: extract "ht" when we have a "___hole rid ht" construct.


rlwrap sertop --printer=human

(Add () "
Definition ___hole {A:Type} (n:nat) (v:A) := v.
Theorem add_easy_0: forall n:nat, 0 + n = n. refine (___hole 0 _). Show Proof.")
(Exec 5)

...

(Parse () "(__hole 0 ?Goal).")

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
                         (line_nb_last 1) (bol_pos_last 0) (bp 10) (ep 15)))))) ))) )
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

...

(Print ((sid 5) (pp ((pp_format PpStr))))
(
(ConstrTerm
                     ((v (CEvar (Id Goal) ()))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 10) (ep 15))))))
)
)


# --

rlwrap sertop --printer=human

(Add () "
Definition ___hole {A:Type} (v:A) := v.
Theorem easy: nat. refine (___hole _). apply O. Show Proof.
")

(Exec 6)

# (Parse () "(__hole 0 ?Goal).")

(Parse () "(___hole 0).")
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
                 ( ((v (Ser_Qualid (DirPath ()) (Id ___hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 8)))))
                  ((ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))))) ) )
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
          (GenArg raw (ExtraArg ltac_use_default) false))))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 12)))))))))
(Answer 2 Completed)


(Print ((sid 6) (pp ((pp_format PpStr))))
(
CoqGenArg
(GenArg raw (ExtraArg tactic)
           (TacArg
            ((v
              (TacCall
               ((v
                 (((v (Ser_Qualid (DirPath ()) (Id ___hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 8)))))
                  ( (ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))) )))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)
)

(Answer 3 Ack)
(Answer 3 (ObjList ((CoqString "___hole 0"))))
(Answer 3 Completed)


# -
# wrap 0 proof term and send it in s-exp format to coq-serapi and have it pprint it back to me

# first I want to see how zero is wrapped up in GenArg from a response. Then perhaps I can map that to the answer I
# already have to figure out how to extract what I need. 

rlwrap sertop --printer=human

(Add () "
Theorem easy: nat. apply O. Show Proof.
")

(Exec 4)

(Parse () "0.")
#doesnt work :/

# - try extract the content of the hole

rlwrap sertop --printer=human

(Add () "
Definition ___hole {A:Type} (v:A) := v.
Theorem easy: nat. refine (___hole _). apply O. Show Proof.
")

(Exec 6)

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
( (ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))) )
)
)
)

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))) )
)
)
)


(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqConstr
(ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))
)

((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqConstr
((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
 (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))
)
)

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqConstr
(CPrim (Numeral SPlus ((int 0) (frac "") (exp ""))))
)
)

(Print ((sid 6) (pp ((pp_format PpStr))))
(
CoqGenArg
(GenArg raw (ExtraArg tactic)
           (TacArg
            ((v
              (TacCall
               ((v
                 ( 
                  ( (ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))) ) ))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10))))) ))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)
)


(Print ((sid 6) (pp ((pp_format PpStr))))
(
CoqGenArg
(GenArg raw (ExtraArg tactic)
           (TacArg
            ((v
              (TacCall
               ((v
                 ( ()
                  ( (ConstrMayEval
                    (ConstrTerm
                     ((v
                       (CPrim (Numeral SPlus ((int 0) (frac "") (exp "")))))
                      (loc
                       (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                         (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))) ) ))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10))))) ))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)
)


(Parse () "0.")
(Answer 2 Ack)
(Answer 2
 (CoqExn
  ((loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
      (bol_pos_last 0) (bp 1) (ep 2))))
   (stm_ids ()) (backtrace (Backtrace ()))
   (exn
    (Stream.Error
     "':' expected after [selector_body] (in [vernac:toplevel_selector])"))
   (pp
    (Pp_box (Pp_hovbox 0)
     (Pp_glue
      ((Pp_string "Syntax error: ")
       (Pp_string
        "':' expected after [selector_body] (in [vernac:toplevel_selector])")
       (Pp_string .)))))
   (str
    "Syntax error: ':' expected after [selector_body] (in [vernac:toplevel_selector])."))))
(Answer 2 Completed)

(Parse () "(___hole O).")
(Answer 3 Ack)
(Answer 3
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
                 (((v (Ser_Qualid (DirPath ()) (Id ___hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 8)))))
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) )) ))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
          (GenArg raw (ExtraArg ltac_use_default) false))))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 12)))))))))
(Answer 3 Completed)

(Parse () "O.")
(Answer 7 Ack)
(Answer 7
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
                 (((v (Ser_Qualid (DirPath ()) (Id O)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 0) (ep 1)))))
                  ()))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 0) (ep 1)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 0) (ep 1)))))))
          (GenArg raw (ExtraArg ltac_use_default) false))))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 2)))))))))
(Answer 7 Completed)

# --

((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) ))

(Print ((sid 6) (pp ((pp_format PpStr))))  
(CoqQualId
(
((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))
)
)

)

(Print ((sid 6) (pp ((pp_format PpStr))))  
(CoqQualId
(Ser_Qualid (DirPath ()) (Id O))
)
)

(Print ((sid 6) (pp ((pp_format PpStr))))  
(CoqQualId
(Qualid (DirPath ()) (Id O))
)
)

(Print ((sid 6) (pp ((pp_format PpStr))))  
(CoqGenArg
(Qualid (DirPath ()) (Id O))
)
)

#--

(Print ((sid 6) (pp ((pp_format PpStr)))) coq_obj)

(Print ((sid 6) (pp ((pp_format PpStr))))
(
CoqGenArg
(GenArg raw (ExtraArg tactic)
           (TacArg
            ((v
              (TacCall
               ((v
                 (((v (Ser_Qualid (DirPath ()) (Id ___hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 8)))))
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) )) ))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)
)

((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) ))

(Print ((sid 6) (pp ((pp_format PpStr)))) (CoqGenArg (...)))

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(CoqGenArg 

(GenArg raw (ExtraArg tactic)
           (TacArg
            ((v
              (TacCall
               ((v
                 (
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) )) ))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10))))))
)

)
)

(Print ((sid 6) (pp ((pp_format PpStr)))) (CoqGenArg (...)))

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
(Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) )

)
)
)

)


(Print ((sid 6) (pp ((pp_format PpStr)))) 
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
((v
                 (
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) ) ) )
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))
)
)
)

)
)

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(CoqGenArg
(GenArg raw (ExtraArg tactic)

(Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) )

)
)
)

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
((v
                 (
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) ) ) )
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))
)
)
)

)
)

# -

(Parse () "(___hole O).")
(Parse () "S S O .")
(Parse () "(S (S (O))).")
(Parse () 
"
___hole ( S (___hole ( S (___hole O)))).
"
)

(Parse () 
"
Definition id := fun (x: nat) => x.
"
)


(Parse () 
"
fun (x: nat) => x.
"
)

(Parse () 
"(___hole (fun (x: nat) => ___hole (x)))."
)

(Parse () "S.")

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(GenArg raw (ExtraArg tactic)
 (TacArg
  ((v
    ((Reference
          ((v (Ser_Qualid (DirPath ()) (Id O)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
              (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))))
   (loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)


(Print ((sid 6) (pp ((pp_format PpStr)))) 
(GenArg raw (ExtraArg tactic)
 (TacArg
  ((v
    (Reference
          ((v (Ser_Qualid (DirPath ()) (Id O)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
              (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))))
   (loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(GenArg raw (ExtraArg tactic)
 (TacArg
  ((v
    ((Reference
          ((v (Ser_Qualid (DirPath ()) (Id O)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
              (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))))
   (loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)

(Print ((sid 6) (pp ((pp_format PpStr)))) 

(GenArg raw (ExtraArg tactic)
 (TacArg
  ((v
    (Reference
          ((v (Ser_Qualid (DirPath ()) (Id O)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
              (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))))
   (loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))

)

# --

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
 (TacArg
  ((v
    (Reference
          ((v (Ser_Qualid (DirPath ()) (Id O)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
              (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))))
   (loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))))
)
)

# --

(Print ((sid 6) (pp ((pp_format PpStr)))) 
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
((v
                 
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))) ) ) 
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))
)
)
)

)
)

# --

(Parse () "(___hole O).")
(Answer 24 Ack)
(Answer 24
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
                 (((v (Ser_Qualid (DirPath ()) (Id ___hole)))
                   (loc
                    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 8)))))
                  ((Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))))))
                (loc
                 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))


))
             (loc
              (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))
            )
          )
          (GenArg raw (ExtraArg ltac_use_default) false))))))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0) (line_nb_last 1)
        (bol_pos_last 0) (bp 0) (ep 12)))))))))
(Answer 24 Completed)


(Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))


(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
(Reference
                    ((v (Ser_Qualid (DirPath ()) (Id O)))
                     (loc
                      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
                        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))
)

)
)
)
)

# ----

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
 (TacArg
  ((v
    (Reference
          ((v (Ser_Qualid (DirPath ()) (Id O)))
           (loc
            (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
              (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10)))))))
   (loc
    (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
      (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10))))))
)
)
)

# ----

# get from the inner TacCall the argument for 0. 

((v
(Reference
    ((v (Ser_Qualid (DirPath ()) (Id O)))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))
)
(loc
 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))

(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
((v
(Reference
    ((v (Ser_Qualid (DirPath ()) (Id O)))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))
)
(loc
 (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
   (line_nb_last 1) (bol_pos_last 0) (bp 1) (ep 10)))))
)
)
)
)


(Print ((sid 6) (pp ((pp_format PpStr))))
(CoqGenArg
(GenArg raw (ExtraArg tactic)
(TacArg
((v
(Reference
    ((v (Ser_Qualid (DirPath ()) (Id O)))
     (loc
      (((fname ToplevelInput) (line_nb 1) (bol_pos 0)
        (line_nb_last 1) (bol_pos_last 0) (bp 9) (ep 10))))))
)
(loc
 ()
)
)
)
)))


# --

```bash
## -- setup opam like VP's PyCoq
#RUN opam init --disable-sandboxing
## compiler + '_' + coq_serapi + '.' + coq_serapi_pin
#RUN opam switch create ocaml-variants.4.07.1+flambda_coq-serapi.8.11.0+0.11.1 ocaml-variants.4.07.1+flambda
#RUN opam switch ocaml-variants.4.07.1+flambda_coq-serapi.8.11.0+0.11.1
#RUN eval $(opam env)
#
#RUN opam repo add coq-released https://coq.inria.fr/opam/released
## RUN opam pin add -y coq 8.11.0
## ['opam', 'repo', '--all-switches', 'add', '--set-default', 'coq-released', 'https://coq.inria.fr/opam/released']
#RUN opam repo --all-switches add --set-default coq-released https://coq.inria.fr/opam/released
#RUN opam update --all
opam pin add -y coq 8.11.0

#RUN opam install -y --switch ocaml-variants.4.07.1+flambda_coq-serapi_coq-serapi_8.11.0+0.11.1 coq-serapi 8.11.0+0.11.1
opam install -y coq-serapi
eval $(opam env)
```
get the serapi top:
```bash
rlwrap sertop --printer=human
```


```
(Add () "
Definition ___hole {A:Type} (v:A) := v.
Theorem easy: nat. refine (___hole _). apply O. Show Proof.
")

(Exec 6)


(Parse () "(___hole O).")
(Parse () "O.")

(Parse () "S S O .")
(Parse () "___hole ( S (___hole ( S (___hole O)))). ")
(Parse () "(___hole (fun (x: nat) => ___hole (x))).")
(Parse () "S.")
(Parse () "(__hole 0 ?Goal).")
(Parse () "Definition id := (fun n : nat => eq_refl).")

(Print ((sid 6) (pp ((pp_format PpStr)))) coq_obj)
```


# when are u in a proof? 

case 1: only declared proof
```
rlwrap sertop --printer=human

(Add () "
Theorem easy: nat -> nat.
")

(Exec 2)

(Query ((pp ((pp_format PpStr)))) Goals)
```
(Answer 2 Ack)
(Answer 2
 (ObjList ((CoqString  "none\
                      \n============================\
                      \nnat -> nat"))))
(Answer 2 Completed)
-> Conclusion: proof state with (lots) of string content, so your in proof mode

case 2: in the middle of some step.
```
rlwrap sertop --printer=human

(Add () "
Theorem easy: nat -> nat. intros.
")

(Exec 3)

(Query ((pp ((pp_format PpStr)))) Goals)
```
(Query ((pp ((pp_format PpStr)))) Goals)
(Answer 2 Ack)
(Answer 2
 (ObjList ((CoqString  "\
                      \n  H : nat\
                      \n============================\
                      \nnat"))))
(Answer 2 Completed)
-> Conclusion: proof state with (lots) of string content, so your in proof mode

case 3: proof is done (but without a Qed.)
```
rlwrap sertop --printer=human

(Add () "
Theorem easy: nat -> nat. intros. apply O.
")
(Exec 4)

(Query ((pp ((pp_format PpStr)))) Goals)
```
(Query ((pp ((pp_format PpStr)))) Goals)
(Answer 2 Ack)
(Answer 2 (ObjList ((CoqString ""))))
(Answer 2 Completed)
-> Conclusion: when proof is done then we have goals being the empty string (note proof term is completed).

case 4: proof is closed (i.e. Qed. like ststement has been called).
```
rlwrap sertop --printer=human

(Add () "
Theorem easy: nat -> nat. intros. apply O. Qed.
")

(Exec 5)

(Query ((pp ((pp_format PpStr)))) Goals)
```
(Query ((pp ((pp_format PpStr)))) Goals)
(Answer 2 Ack)
(Answer 2 (ObjList ()))
(Answer 2 Completed)
-> Conclusion: when proof is closed, then the goals is literally empty, no coq string object. 

# --

rlwrap sertop --printer=human

(Add ()
"
Fixpoint eqb (n m : nat) : bool :=
  match n with
  | O => match m with
         | O => true
         | S m' => false
         end
  | S n' => match m with
            | O => false
            | S m' => eqb n' m'
            end
  end.

Theorem eqb_refl: forall n, eqb n n = true.
Proof. induction n as [|n1 IHn1].
       - simpl. reflexivity.
       - simpl. rewrite -> IHn1. reflexivity.
"
)

(Exec 12)

(Add () "Show Proof.")
(Exec 13)

# -- 

```
Show Conjectures. is how you get the theorem name you are currently trying to prove.  
Because of nested proofs, I guess it theoretically can return more than one thing. 
I suggest adding this to your data if for nothing else than debugging. 
It is easier to look up a theorem by name in a file than a position index. (edited) 
```

rlwrap sertop --printer=human

(Add () "
Theorem add_easy_0'':
forall n:nat,
  0 + n = n.
Proof.
    intros.
    simpl.
")
(Exec 5)

(Add () "Show Conjectures.")
(Exec 6)

(Exec 6)
(Answer 4 Ack)
(Feedback
 ((doc_id 0) (span_id 6) (route 0) (contents (ProcessingIn master))))
(Feedback ((doc_id 0) (span_id 5) (route 0) (contents Processed)))
(Feedback
 ((doc_id 0) (span_id 6) (route 0)
  (contents
   (Message (level Notice) (loc ()) (pp (Pp_string add_easy_0''))
    (str add_easy_0'')))))
(Feedback ((doc_id 0) (span_id 6) (route 0) (contents Processed)))
(Answer 4 Completed)


```
Show Existentials

Displays all open goals / existential variables in the current proof along with the type and the context of each variable.

https://coq.github.io/doc/v8.10/refman/proof-engine/proof-handling.html#coq:cmdv.show-conjectures
3:14
Show Existentials. is interesting too.  It shows the proof state for each goal/e-variable, and also makes clear what the e-variables names are in the proof.
3:15
https://coq.github.io/doc/v8.10/refman/proof-engine/proof-handling.html#coq:cmdv.show-existentials
```

```
Theorem zero_plus_n_eq_n:
  forall n: nat, 0 + n = n.
  Show.
 Proof.
  intros.
  simpl.
  Show Existentials.

Existential 1 =
?Goal : [n : nat |- n = n]

Displays all open goals / existential variables in the current proof along with the type and the context of each variable.
```

# ----

rlwrap sertop --printer=human

(Add () "
Theorem add_easy_induct_1:
forall n:nat,
  n + 0 = n.
Proof.
  intros.
  induction n as [| n' IH].
")
(Exec 5)
(Query ((pp ((pp_format PpStr)))) Goals)


https://github.com/ejgallego/coq-serapi/issues/280

# ----

docker run -v /Users/brandomiranda/iit-term-synthesis:/home/bot/iit-term-synthesis \
           -v /Users/brandomiranda/pycoq:/home/bot/pycoq \
           -v /Users/brandomiranda/ultimate-utils:/home/bot/ultimate-utils \
           -v /Users/brandomiranda/proverbot9001:/home/bot/proverbot9001 \
           -v /Users/brandomiranda/data:/home/bot/data \
           -ti brandojazz/iit-term-synthesis:test_arm bash

rlwrap sertop --printer=human

(Add () "
Theorem add_easy_induct_1:
forall n:nat,
  n + 0 = n.
Proof.
  intros.
  induction n as [| n' IH].
  - simpl.
    reflexivity.
  - simpl.
    rewrite -> IH.
    reflexivity.
")
(Exec 12)
(Query ((pp ((pp_format PpStr)))) Goals)

rlwrap sertop --printer=human

(Add () "
Theorem add_easy_induct_1:
forall n:nat,
  n + 0 = n.
Proof.
  intros.
  induction n as [| n' IH].
  - simpl.
    reflexivity.
  - simpl.
    rewrite -> IH.
    reflexivity.
Qed.
")
(Exec 13)
(Query ((pp ((pp_format PpStr)))) Goals)

python -m pdb -c continue ~/iit-term-synthesis/iit-term-synthesis-src/data_pkg/data_gen.py
