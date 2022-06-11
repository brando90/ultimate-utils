# Tutorial 1

## Recall [optional]
In case it helps, recall tutorial 0:
```makefile
targets: prerequisites
	command
	command
	command
```
the target is the rule and the file name to build tha files. 
Beneath are the command to build that file and the prerequisites are other targets/rules that 
need to be run (or at least their files need to be up to date) before the commands specified for this
target is ran.
Note, that of course the rules/target don't correspond to any file if the target/rule name is 
in the `.PHONY` variable -- e.g. `.PHONY := clean`.

## Building actual files

Consider the Makefile:
```makefile
blah: blah.o
	cc blah.o -o blah # Runs third

blah.o: blah.c
	cc -c blah.c -o blah.o # Runs second

blah.c:
	echo "int main() { return 0; }" > blah.c # Runs first
```
if you run `make` it runs the first rule (I assume no arg is given or no `DEFAULT_GOAL` is specified) 
and it looks to see if all the dependencies/prerequisites are up to date (including itself i.e. if the `blah` file exists). 
If any of the prerequisites is not up-to-date (e.g. doesn't exist or the current files is older than an edit to
something in the prerequisites) then it rebuilds the prerequisites (i.e. runs the commands in the targets in the prerequisites)
and then re-builds the current target that was called (in this case the one at the top of the file).

First time this is run it creates everyting:
```makefile
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ ls
Makefile        readme.md
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ make 
echo "int main() { return 0; }" > blah.c # Runs first
cc -c blah.c -o blah.o # Runs second
cc blah.o -o blah # Runs third
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ ls
Makefile        blah            blah.c          blah.o          readme.md
```


Second time we run it let's pretend that `blah.c` is updated. e.g. we manually change it.
What will happen is that `make` will detect a change there but since the file already exists it won't
run the "rebuild" commands under the `blah.c` target. However, since the previous files depended on it 
-- since they were prerequisites -- what happens is that all of those are re build.
In particular, note that the:
```makefile
echo "int main() { return 0; }" > blah.c
```
is never ran.
See terminal:
```makefile
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ ls
Makefile        blah            blah.c          blah.o          readme.md
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ cat blah.c
int main() { return 0; }
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ echo "int main() { return 1; }" > blah.c
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ cat blah.c
int main() { return 1; }
(iit_synthesis) brandomiranda~/ultimate-utils/tutorials_for_myself/my_makefile_playground/beginner_makefile_tutorial1 ❯ make
cc -c blah.c -o blah.o # Runs second
cc blah.o -o blah # Runs third
```

Main take away:
- targets are rules that usually generate files. If the file already exists its commands to rebuild it under the rule will not be re-ran.
- targets have prerequisites/dependencies that are other rules. If one of those prereq rule gets updated then we have
  to rebuild all the targets that depeended on it. So if something at the bottom of the dag is updated (or doesn't)
  exists -- then everything above it (that depends on it) is rebuilt.

## Exmphasis on .PHONY

Names in `.PHONY` won't be associated to a file so those rules usually always run 
(or some ppl say "build" or "rebuild" but I think that is bad naming because the target is not assciated
to a file anymore so running a command is a better way to speak imho).
E.g. e.g. `.PHONY := clean` likely means that the `clean` rule is always ran which usually deletes files
in the current directory.

## Ref

- https://makefiletutorial.com/
- https://www.youtube.com/watch?v=zeEMISsjO38