# Tutorial 0

Make builds/compiles files based on dependencies.
In this example lets consider:
```makefile
another:
	echo "hello"
```
if you run the `make` command (or `make another`) what it actually does is
look for the file `another` (in the current dir I think or it likely assumes that since the target has no path in the name? idk) 
and if it does not exist then it runs the 
commands bellow the `another` rule (also called target).
Implicitly this assumes that `another` is a rule/target to build a file using
the commands bellow the rule. 
In this case it doesn't actually build anything every by running `make another`
but if the rule create `another` then it would.

See interation:
```makefile
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ ls
Makefile        readme.md
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ make another
echo "hello"
hello
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ make another
echo "hello"
hello
(iit_synth
```

Now let's see if there was an another file. What happens? What if we delete this another file later and run `make another`?
```makefile
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ ls
Makefile        readme.md
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ make another
echo "hello"
hello
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ touch another
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ make another
make: `another' is up to date.
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ make another
make: `another' is up to date.
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ rm another
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial0 ❯ make another
echo "hello"
hello
```

Main take away:
- make is used to build files.
- Targets are the rules in a make file ran if the files or its dependencies are not up to date and runs the command bellow it. 
 It assumes the commands bellow a rule/target are the cmds to generate/build/make the file.

## Ref

- https://makefiletutorial.com/