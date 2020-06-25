def test_episodic_loader_inner_loop_per_task(debug_test=True):
    import automl.child_models.learner_from_opt_as_few_shot_paper as learner_from_opt_as_few_shot_paper
    import higher
    
    ## get args for test
    args = get_args_for_mini_imagenet()
    ## get base model that meta-lstm/maml
    base_model = learner_from_opt_as_few_shot_paper.Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)
    ## get meta-sets
    metatrainset_loader, metavalset_loader, metatestset_loader = prepare_data_for_few_shot_learning(args)
    ## start episodic training
    meta_params = base_model.parameters()
    outer_opt = optim.Adam(meta_params, lr=1e-3)
    base_model.train()
    # sample a joint set SQ of k_shot+k_eval examples
    for episode, (SQ_x, SQ_y) in enumerate(metatrainset_loader):
        ## Sample the support S & query Q data e.g. S = {S_t}^N_t, Q = {Q_t}^N_t
        S_x, S_y, Q_x, Q_y = get_support_query_batch_of_tasks_class_is_task_M_eq_N(args, SQ_x, SQ_y)
        ## Get Inner Optimizer (for maml)
        inner_opt = torch.optim.SGD(base_model.parameters(), lr=1e-1)
        nb_tasks = S_x.size(0) # extract N (=M) tasks note: torch.Size([N, k_shot+k_eval, C, H, W])
        with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            meta_loss = 0 # computes 1/M \sum^M_t L(A(\theta,S_t), Q_t)
            for t in range(nb_tasks):
                ## Inner-Adaptation Loop for the current task i.e. \theta^<i_inner+1> := \theta^<t_Outer,T> - eta_inner * \grad _{\theta} L(\theta^{<t_Outer,t_inner>},S_t)
                # sample current task s.t. support data is aligned with corresponding query data
                Sx_t, Sy_t, Qx_t, Qy_t = S_x[t,:,:,:], S_y[t,:], Q_x[t,:,:,:], Q_y[t,:]
                # Inner-Adaptation Loop for the current task i.e. \theta^<i_inner+1> := \theta^<t_Outer,T> - eta_inner * \grad _{\theta} L(\theta^{<t_Outer,t_inner>},S_t)
                # note that we could train here in batches from S_t but since S_t is so small k_shot (1 or 5) for each class/task t \in [N], we use the whole thing
                for i_inner in range(args.nb_inner_train_steps): # this current version implements full gradient descent on k_shot examples (which is usually small  5)
                    fmodel.train()
                    # base/child model forward pass
                    S_logits_t = fmodel(Sx_t) 
                    inner_loss = args.criterion(S_logits_t, Sy_t)
                    # inner-opt update
                    diffopt.step(inner_loss)
                ## Evaluate on query set for current task
                qrt_logits_t = fmodel(Qx_t)
                meta_loss += args.criterion(qrt_logits_t, Qy_t)
            meta_loss = meta_loss / nb_tasks
        print(f'[episode={episode}] base_model.model.features.conv1.weight.grad = {base_model.model.features.conv1.weight.grad}')
        meta_loss.backward()
        outer_opt.step()
        outer_opt.zero_grad()
        print(f'[episode={episode}] meta_loss = {meta_loss}')