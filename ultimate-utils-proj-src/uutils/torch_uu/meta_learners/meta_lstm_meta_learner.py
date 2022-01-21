import torch
import torch.nn as nn

import higher

from uutils.torch_uu import calc_accuracy
from uutils.torch_uu import preprocess_grad_loss


####
####
from uutils.torch_uu.models.custom_layers import Flatten


class EmptyMetaLstmOptimizer(Optimizer):

    def __init__(self, params, *args, **kwargs):
        defaults = {'args':args, 'kwargs':kwargs}
        super().__init__(params, defaults)

class MetaTrainableLstmOptimizer(DifferentiableOptimizer):
    '''
    Adapted lstm-meta trainer from Optimization as a model for few shot learning.
    '''

    def _update(self, grouped_grads, **kwargs):
        ## unpack params to update
        trainable_opt_model = self.trainable_opt_model
        trainable_opt_state = self.param_groups[0]['kwargs']['trainable_opt_state']
        [(lstmh, lstmc), metalstm_hx] = trainable_opt_state['prev_state']
        inner_train_loss = self.param_groups[0]['kwargs']['trainfo_kwargs']['inner_train_loss']
        inner_train_err = self.param_groups[0]['kwargs']['trainfo_kwargs']['inner_train_err']
        # get the flatten params & grads
        zipped = zip(self.param_groups, grouped_grads)
        flatten_params = []
        flatten_grads = []
        flatten_lengths = []
        original_sizes = []
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                original_sizes.append(p.size())
                g = g.view(-1) # flatten vector
                p = p.view(-1) # flatten vector
                assert( len(g.size())==1 and len(p.size())==1 )
                assert( g.size(0)== p.size(0) )
                flatten_lengths.append( p.size(0) )
                flatten_grads.append(g)
                flatten_params.append(p)
        flatten_params = torch.cat(flatten_params, dim=0)
        flatten_grads = torch.cat(flatten_grads, dim=0)
        n_learner_params = flatten_params.size(0)
        # hx i.e. previous forget, update & cell state from metalstm
        if None in metalstm_hx:
            # set initial f_prev, i_prev, c_prev]
            metalstm_hx = trainable_opt_model.initialize_meta_lstm_cell(flatten_params)
        # preprocess grads
        grad_prep = preprocess_grad_loss(flatten_grads)  # [|theta_p|, 2]
        loss_prep = preprocess_grad_loss(inner_train_loss) # [1, 2]
        err_prep = preprocess_grad_loss(inner_train_err) # [1, 2]
        # get new parameters from meta-learner/trainable optimizer/meta-lstm optimizer
        theta_next, [(lstmh, lstmc), metalstm_hsx] = trainable_opt_model(
            inputs=[loss_prep, grad_prep, err_prep, flatten_grads],
            hs=[(lstmh, lstmc), metalstm_hx]
        )
        #assert( theta_next )
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        i = 0
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                ## update params with new theta: f^<t>*theta^<t-1> - i^<t>*grad^<t>
                original_size = original_sizes[p_idx]
                p_len = flatten_lengths[p_idx]
                assert(p_len == np.prod(original_size))
                p_new  = theta_next[i:i+p_len]
                p_new = p_new.view(original_size)
                group['params'][p_idx] = p_new
                i = i+p_len
        assert(i == n_learner_params)
        # fake returns
        self.param_groups[0]['kwargs']['trainable_opt_state']['prev_lstm_state'] = [(lstmh, lstmc), metalstm_hx]

class MetaLSTMCell(nn.Module):
    """
    Based on: https://github.com/brando90/meta-learning-lstm-pytorch
    Or: https://github.com/markdtw/meta-learning-lstm-pytorch

    Model:
    C_t = f_t * C_{t-1} + i_t * \tilde{C_t}
    """
    def __init__(self, device, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size # equal to first layer lstm.hidden_size e.g. 32
        self.hidden_size = hidden_size
        assert(self.hidden_size == 1)
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size)) # [input_size+2, 1]
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size)) # [input_size+2, 1]
        #self.cI = nn.Parameter(torch_uu.Tensor(n_learner_params, 1)) # not needed because training init is handled somewhere ese
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        ## reset all the params of meta-lstm trainer (including cI init of the base/child net if they are included in the constructor)
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        ## want initial forget value to be high and input value to be low so that model starts with gradient descent
        ## f^<t>*c^<t-1> + i^<t>*c~^<t> =~= theta^<t-1> - i^<t> * grad^<t>
        # set unitial forget value to be high so that f^<t>*theta^<t> = 1*theta^<t>
        nn.init.uniform_(self.bF, a=4, b=6) # samples from Uniform(a,b)
        # set initial learning rate to be low so its approx GD with small step size at start
        nn.init.uniform_(self.bI, -5, -4) # samples from Uniform(a,b)

    def init_cI(self, flat_params):
        self.cI.data.copy_(flat_params.unsqueeze(1))

    def forward(self, inputs, hx=None):
        # gunpack inputs
        lstmh, grad = inputs # (lstm(grad_t, loss_t), grad)
        n_learner_params, input_size = lstmh.size(1), lstmh.size(2) # note input_size = hidden_size for first layer lstm. Hidden size for meta-lstm learner is 1.
        grad = grad.view(n_learner_params, 1) # dim -> [n_learner_params, 1]
        f_prev, i_prev, c_prev = hx
        # change dim so matrix mult mm work: dim -> [n_learner_params, 1]
        lstmh = lstmh.view(n_learner_params, input_size) # -> [input_size x n_learner_params] = [hidden_size x n_learner_params]
        f_prev, i_prev, c_prev = f_prev.view(n_learner_params,1), i_prev.view(n_learner_params,1), c_prev.view(n_learner_params,1)
        # f_t = sigmoid(W_f * [ lstm(grad_t, loss_t), theta_{t-1}, f_{t-1}] + b_f)
        f_next = sigmoid( cat((lstmh, c_prev, f_prev), 1).mm(self.WF) + self.bF.expand_as(f_prev) ) # [n_learner_params, 1] = [n_learner_params, input_size+3] x [input_size+3, 1] + [n_learner_params, 1]
        # i_t = sigmoid(W_i * [ lstm(grad_t, loss_t), theta_{t-1}, i_{t-1}] + b_i)
        i_next = sigmoid( cat((lstmh, c_prev, i_prev), 1).mm(self.WI) + self.bI.expand_as(i_prev) ) # [n_learner_params, 1] = [n_learner_params, input_size+3] x [input_size+3, 1] + [n_learner_params, 1]
        # next cell/params: theta^<t> = f^<t>*theta^<t-1> - i^<t>*grad^<t>
        c_next = f_next*(c_prev) - i_next*(grad) # note, - sign is important cuz i_next is positive due to sigmoid activation

        # c_next.squeeze() left for legacydsgdfagsdhsjsjhdfhjdhgjfghjdgj
        #c_next = c_next.squeeze()
        assert(c_next.size() == torch.Size((n_learner_params,1)))
        return c_next.squeeze(), [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)

class MetaLstmOptimizer(nn.Module):
    '''
    Meta-learner/optimizer based on Optimization as a model for few shot learning.
    '''

    def __init__(self, device, input_size, hidden_size, num_layers=1):
        """Args:
            input_size (int): for the first LSTM layer, 6
            hidden_size (int): for the first LSTM layer, e.g. 32
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers
        ).to(device)
        #self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size).to(device)
        self.metalstm = MetaLSTMCell(device=device, input_size=hidden_size, hidden_size=1).to(device)
        self.to(device)

    def forward(self, inputs, hs=None):
        loss_prep, grad_prep, err_prep, grad = inputs
        ## forward pass of first lstm
        # sort out input x^<t> to normal lstm
        loss_prep = loss_prep.expand_as(grad_prep) # [1, 2] -> [n_learner_params, 2]
        err_prep = err_prep.expand_as(grad_prep) # [1, 2] -> [n_learner_params, 2]
        xn_lstm = torch.cat((loss_prep, err_prep, grad_prep), 1).unsqueeze(0) # [n_learner_params, 6]
        # normal lstm([loss, grad_prep, train_err]) = lstm(xn)
        n_learner_params = xn_lstm.size(1)
        (lstmh, lstmc) = hs[0] # previous hx from first (standard) lstm i.e. lstm_hx = (lstmh, lstmc) = hs[0]
        if lstmh.size(1) != xn_lstm.size(1): # only true when prev lstm_hx is equal to decoder/controllers hx
            # make sure that h, c from decoder/controller has the right size to go into the meta-optimizer
            expand_size = torch.Size([1,n_learner_params,self.lstm.hidden_size])
            lstmh, lstmc = lstmh.squeeze(0).expand(expand_size).contiguous(), lstmc.squeeze(0).expand(expand_size).contiguous()
        lstm_out, (lstmh, lstmc) = self.lstm(input=xn_lstm, hx=(lstmh, lstmc))

        ## forward pass of meta-lstm i.e. theta^<t> = f^<t>*theta^<t-1> + i^<t>*grad^<t>
        metalstm_hx = hs[1] # previous hx from optimizer/meta lstm = [metalstm_fn, metalstm_in, metalstm_cn]
        xn_metalstm = [lstmh, grad] # note, the losses,grads are preprocessed by the lstm first before passing to metalstm [outputs_of_lstm, grad] = [ lstm(losses, grad_preps), grad]
        theta_next, metalstm_hx = self.metalstm(inputs=xn_metalstm, hx=metalstm_hx)

        return theta_next, [(lstmh, lstmc), metalstm_hx]

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        inner_opt, args = kwargs['inner_opt'], kwargs['args']
        # process hidden state from arch decoder/controller
        h = ( ( out.mean()+h )/2 ).expand_as(h)
        # if lstmh.size() != xn_lstm.size():
        #     lstmh = torch_uu.cat((lstmh,lstmh,lstmh),dim=2).squeeze(0).expand_as(xn_lstm)
        #     lstmc = torch_uu.cat((lstmc,lstmc,lstmc),dim=2).squeeze(0).expand_as(xn_lstm)
        trainable_opt_state = {}
        # reset h, c to be the decoders out & h
        [(lstmh, lstmc), metalstm_hx] = [(h, c), (None, None, None)]
        if inner_opt is None: # inner optimizer has not been used yet
            pass
        else: # use info from a the inner optimizer from previous outer loop
            # if you want to use the prev inner optimizer's & decoder/controler's (h, c)
            #h, c = (h + h_opt.detach())/2, (c + c_opt.detach())/2
            pass
        trainable_opt_state['prev_state'] = [(lstmh, lstmc), metalstm_hx]
        ## create initial trainable opt state
        return trainable_opt_state

    def initialize_meta_lstm_cell(self, flatten_params):
        device = flatten_params.device
        n_learner_params = flatten_params.size(0)
        f_prev = torch.zeros((n_learner_params, self.metalstm.hidden_size)).to(device)
        i_prev = torch.zeros((n_learner_params, self.metalstm.hidden_size)).to(device)
        c_prev = flatten_params
        meta_hx = [f_prev, i_prev, c_prev]
        return meta_hx

higher.register_optim(EmptyMetaLstmOptimizer, MetaTrainableLstmOptimizer)

#### tests
####

def test_good_accumulator_simple(verbose=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from collections import OrderedDict

    from types import SimpleNamespace

    ## training config
    args = SimpleNamespace()
    #args.device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")
    args.mode = 'meta-train'
    #args.track_higher_grads = True # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    args.copy_initial_weights = False # if False then we train the base base_models initial weights (i.e. the base model's initialization)
    args.episodes = 5
    args.nb_inner_train_steps = 1
    args.outer_lr = 1e-2
    args.inner_lr = 1e-1 # carefuly if you don't know how to change this one
    # N-way, K-shot, with k_eval points
    args.k_shot, args.k_eval = 5, 15
    args.n_classes = 2
    D = 1
    # loss for tasks
    args.criterion = nn.CrossEntropyLoss() # The input is expected to contain raw, unnormalized scores for each class.
    ## get base model
    nb_hidden_units = 1
    base_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Linear(D, nb_hidden_units)),
        ('act', nn.LeakyReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear(nb_hidden_units, args.n_classes))
        ]))
    meta_params = base_model.parameters()
    outer_opt = optim.Adam(meta_params, lr=args.outer_lr)
    for episode in range(args.episodes):
        ## get fake support & query data from batch of N tasks
        spt_x, spt_y = torch.randn(args.n_classes, args.k_shot, D), torch.stack([ torch.tensor([label]).repeat(args.k_shot) for label in range(args.n_classes) ])
        qry_x, qry_y = torch.randn(args.n_classes, args.k_eval, D), torch.stack([ torch.tensor([label]).repeat(args.k_eval) for label in range(args.n_classes) ])
        ## Compute grad Meta-Loss
        inner_opt = torch.optim.SGD(base_model.parameters(), lr=args.inner_lr)
        nb_tasks = spt_x.size(0) # extract N tasks. Note M=N
        meta_losses = []
        for t in range(nb_tasks):
            with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=args.copy_initial_weights) as (fmodel, diffopt):
                spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t,:,:], spt_y[t,:], qry_x[t,:,:], qry_y[t,:]
                for i_inner in range(args.nb_inner_train_steps):
                    fmodel.train()
                    # base/child model forward pass
                    S_logits_t = fmodel(spt_x_t) 
                    inner_loss = args.criterion(S_logits_t, spt_y_t)
                    # inner-opt update
                    diffopt.step(inner_loss)
                ## Evaluate on query set for current task
                qrt_logits_t = fmodel(qry_x_t)
                qrt_loss_t = args.criterion(qrt_logits_t,  qry_y_t)
                ## Backard grad accumulator
                qrt_loss_t.backward() # this accumualtes the gradients in a memory efficient way to compute the desired gradients on the meta-loss for each task
        if verbose:
            print(f'--> episode = {episode}')
            print(f'meta-loss = {sum(losess)}, meta-accs = {sum(meta_accs)}')
            print(f'base_model.conv1.weight.grad= {base_model.conv1.weight.grad}')
            print(f'base_model.conv1.bias.grad = {base_model.conv1.bias.grad}')
            print(f'base_model.fc1.weight.grad = {base_model.fc1.weight.grad}')
            print(f'base_model.fc1.bias.grad = {base_model.fc1.bias.grad}')
        assert(base_model.conv1.weight.grad is not None)
        assert(base_model.fc1.weight.grad is not None)
        outer_opt.step()
        outer_opt.zero_grad()

if __name__ == "__main__":
    test_good_accumulator_simple()
    print('Done, all Tests Passed! \a')