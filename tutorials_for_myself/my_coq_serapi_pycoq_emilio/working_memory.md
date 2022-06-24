# Crafting printing human friendly
(Print ((sid 4) (pp ((pp_format PpStr)))) (CoqConstr (App (Rel 0) ((Rel 0)))))

(Print  ((sid 4) (pp ((pp_format PpStr))))   (CoqConstr (App (Rel 0) ((Rel 0))))) )

# Get all context as a single string

``` 
(Query ((sid 3) (pp ((pp_format PpStr)))) Goals)
```