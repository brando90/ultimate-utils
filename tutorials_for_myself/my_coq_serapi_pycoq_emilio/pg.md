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
```

