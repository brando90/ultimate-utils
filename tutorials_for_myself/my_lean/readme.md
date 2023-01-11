# Installing

Quick start: https://leanprover.github.io/lean4/doc/quickstart.html (basically install the extension)

For us the terminal didn't know where the bin was so we had to find it at:
```
$HOME/.elan/bin
```
and added this to the `.zshrc` file:
```
export PATH=$HOME/.elan/bin:$PATH
```
Now it should work:
```
(meta_learning) brandomiranda~/ultimate-utils/tutorials_for_myself/my_lean ❯ which lake
/Users/brandomiranda/.elan/bin/lake
(meta_learning) brandomiranda~/ultimate-utils/tutorials_for_myself/my_lean ❯ which lean
/Users/brandomiranda/.elan/bin/lean
```

## Misc

- https://leanprover-community.github.io/install/project.html
