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

```
# (Query (opts) query)
(Query ((pp ((pp_format PpStr)))) Proof)
```

``` 
opam reinstall --yes --switch ocaml-variants.4.07.1+flambda_coq-serapi.8.11.0+0.11.1 --keep-build-dir lf
```


