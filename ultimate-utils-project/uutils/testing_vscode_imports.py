

def test():
    import uutils.emailing
    import uutils # this should not have a yellow line underlying it

    print(uutils)
    print(uutils.helloworld)
    uutils.helloworld()

    ## try to import helloworld2 and do control+space, does the doc show?
    # uutils.helloworld2() # uncomment me and do control space
    # uutils.helloworld2

    from uutils.helloworld as helloworld

    out = uutils.helloworld2

    ###

    from automl.child_models.learner_from_opt_as_few_shot_paper import Learner

    learner = Learner()

if __name__ == '__main__':
    test()