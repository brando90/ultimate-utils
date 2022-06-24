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

```
(Add () "Lemma addn0 n : n + 0 = n.")
(Exec 2)
(Add () "Proof.")
(Exec 3)

# (Query (opts) query)
(Query ((sid 3) (pp ((pp_format PpStr)))) Goals)
(Query ((pp ((pp_format PpStr)))) Goals)
(Query () Goals)

(Print (pp (pp_format PpStr)) Obj ) 
```

What about this algorithm 1 (from VPs to many round tripts to back serapi):

- go through all the parsed terms VP gave us.
- for each term, construct the right serapi-exp (replaced `[]` with `()`)
- test it with the `(Print (pp (pp_format PpStr)) Obj )`. There you go, you have a Coq pretty printed string object.

Note: if you query all goals at once you get one big string.
Is this true with PyCoq-SerApi?

TODO: can this be done with PyCoq-SerApi?





