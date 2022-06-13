# KNOWNTARGETS will not be passed along to CoqMakefile
KNOWNTARGETS := CoqMakefile
# KNOWNFILES will not get implicit targets from the final rule, and so
# depending on them won't invoke the submake. TODO: Not sure what this means
# Warning: These files get declared as PHONY, so any targets depending
# on them always get rebuilt -- or better said, any rules which those names will have their cmds be re-ran (which is
# usually rebuilding functions since that is what make files are for)
KNOWNFILES := Makefile _CoqProject

# Runs invoke-coqmakefile rule if you do run make by itself. Sets the default goal to be used if no targets were specified on the command line.
.DEFAULT_GOAL := invoke-coqmakefile

# Depends on two files to run this, itself and our _CoqProject
CoqMakefile: Makefile _CoqProject
	$(COQBIN)coq_makefile -f _CoqProject -o CoqMakefile

# Note make knows what is the make file in this case thanks to -f CoqMakefile
invoke-coqmakefile: CoqMakefile
	$(MAKE) --no-print-directory -f CoqMakefile $(filter-out $(KNOWNTARGETS),$(MAKECMDGOALS))

#
.PHONY: invoke-coqmakefile $(KNOWNFILES)

####################################################################
##                      Your targets here                         ##
####################################################################

# This should be the last rule, to handle any targets not declared above
%: invoke-coqmakefile
	@true