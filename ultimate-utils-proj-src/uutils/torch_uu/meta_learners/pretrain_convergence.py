from argparse import Namespace

import torch
import torch.nn as nn

import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from pathlib import Path

from types import SimpleNamespace

from sklearn.metrics import log_loss
from torch import Tensor

from uutils.logger import Logger

# import automl.child_models.learner_from_opt_as_few_shot_paper.Learner

# https://github.com/WangYueFt/rfs/blob/master/eval/meta_eval.py
from uutils.torch_uu import r2_score_from_torch, process_meta_batch, tensorify, normalize
from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval, \
    mean_confidence_interval
from uutils.torch_uu.models import getattr_model

from pdb import set_trace as st, set_trace


class FitFinalLayer(nn.Module):

    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 target_type: str = 'classification',
                 classifier: str = 'LR',
                 ):
        super().__init__()
        self.args = args
        self.model = model
        self.target_type = target_type
        self.classifier = classifier

    @property
    def base_model(self):
        return self.model

    # set field as function base_model
    @base_model.setter
    def base_model(self, model: nn.Module):
        self.model = model

    def forward(self, batch,
                training: bool = True,
                is_norm: bool = False,
                ):
        """
        training true since we want BN to use batch statistics (and not cheat, etc)
        """
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(self.args, batch)
        meta_batch_size = spt_x.size(0)
        # -- Get average meta-loss/acc of the meta-learner i.e. 1/B sum_b Loss(qrt_i, f) = E_B E_K[loss(qrt[b,k], f)]
        # average loss on task of size K-eval over a meta-batch of size B (so B tasks)
        meta_losses, meta_accs = self.get_list_accs_losses(batch, training, is_norm)
        assert (len(meta_losses) == meta_batch_size)

        # -- return loss, acc with CIs
        meta_loss, meta_loss_ci = mean_confidence_interval(meta_losses)
        meta_acc, meta_acc_ci = mean_confidence_interval(meta_accs)
        return meta_loss, meta_loss_ci, meta_acc, meta_acc_ci

    def eval_forward(self, batch,
                     training: bool = True,
                     is_norm: bool = False,
                     ):
        """
        note:
            - Does a forward pass. It's the same as forward just so that all Agents have the same interface.
            This one looks redundant and it is, but it's here for consistency with the SL agents.
            The eval forward is different in SL agents.
            - training true since we want BN to use batch statistics (and not cheat, etc)
        """
        meta_loss, meta_loss_std, meta_acc, meta_acc_std = self.forward(batch, training, is_norm)
        return meta_loss, meta_loss_std, meta_acc, meta_acc_std

    def get_lists_accs_losses(self, batch,
                              training: bool = True,
                              is_norm: bool = False,
                              ):
        """
        Get the list of accuracies and losses for each task in the meta-batch
        """
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(self.args, batch)
        # Accumulate gradient of meta-loss wrt fmodel.param(t=0)
        meta_batch_size = spt_x.size(0)
        meta_losses, meta_accs = [], []
        for t in range(meta_batch_size):
            spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]

            self.model.train() if training else self.model.eval()
            spt_embeddings_t = self.get_embedding(spt_x_t, self.model).detach()
            qry_embeddings_t = self.get_embedding(qry_x_t, self.model).detach()

            if is_norm:
                spt_embeddings_t = normalize(spt_embeddings_t)
                qry_embeddings_t = normalize(qry_embeddings_t)

            if self.target_type == 'classification':
                spt_embeddings_t = spt_embeddings_t.view(spt_embeddings_t.size(0), -1).cpu().numpy()
                qry_embeddings_t = qry_embeddings_t.view(qry_embeddings_t.size(0), -1).cpu().numpy()
                spt_y_t = spt_y_t.view(-1).cpu().numpy()
                qry_y_t = qry_y_t.view(-1).cpu().numpy()
                # print(f'{spt_embeddings_t.shape=}')

                # Inner-Adapt final layer with spt set
                mdl = LogisticRegression(random_state=0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                mdl.fit(spt_embeddings_t, spt_y_t)

                # Predict using adapted final layer with qrt set
                if self.classifier == 'LR':
                    # note C=1.0 **IS** the defualt, same with penalty
                    # clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                    #                          multi_class='multinomial')
                    # original rfs (using original incase API changes defaults)
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(spt_embeddings_t, spt_y_t)
                    query_y_pred_t = clf.predict(qry_embeddings_t)
                    query_y_probs_t = clf.predict_proba(qry_embeddings_t)
                elif self.classifier == 'SVM':
                    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                    #                                           C=1,
                    #                                           kernel='linear',
                    #                                           decision_function_shape='ovr'))
                    # clf.fit(support_features, support_ys)
                    # query_ys_pred = clf.predict(query_features)
                    raise ValueError(f'Not tested {self.classifier}')
                elif self.classifier == 'NN':
                    query_y_pred_t = NN(spt_embeddings_t, spt_y_t, qry_embeddings_t)
                    raise ValueError(f'Not tested {self.classifier}')
                elif self.classifier == 'Cosine':
                    query_y_pred_t = Cosine(spt_embeddings_t, spt_y_t, qry_embeddings_t)
                    raise ValueError(f'Not tested {self.classifier}')
                elif self.classifier == 'Proto':
                    # query_ys_pred = Proto(support_features, support_ys, query_features, opt)
                    raise ValueError(f'Not tested {self.classifier}')
                else:
                    raise NotImplementedError(f'classifier not supported: {self.classifier}')
                # acc
                qry_loss_t = log_loss(qry_y_t, query_y_probs_t)
                qry_acc_t = metrics.accuracy_score(query_y_pred_t, qry_y_t)
            elif self.target_type == 'regression':
                # Inner-Adapt final adapted layer with spt set
                mdl = LinearRegression().fit(spt_embeddings_t, spt_y_t)
                # Predict using adapted final layer with qrt set
                query_y_pred_t = torch.Tensor(mdl.predict(qry_embeddings_t))
                qry_loss_t = self.args.criterion(query_y_pred_t, qry_y_t)
                qry_acc_t = r2_score_from_torch(qry_y_t, query_y_pred_t)
            else:
                raise ValueError(f'Not implement: {self.target_type}')

            # collect losses & accs
            meta_losses.append(qry_loss_t.item())
            meta_accs.append(qry_acc_t)
        return meta_losses, meta_accs

    def get_embedding(self, x: Tensor, model: nn.Module) -> Tensor:
        return get_embedding(x=x, model=model)

    def regression(self):
        self.target_type = 'regression'

    def classification(self):
        self.target_type = 'classification'

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


def get_adapted_according_to_ffl(model, spt_x_t, spt_y_t, qry_x_t, qry_y_t,
                                 layer_to_replace: str,
                                 training: bool = True,
                                 target_type: str = 'classification',
                                 classifier: str = 'LR', ) -> nn.Module:
    """
    Return the adapted model such that the final layer has the LR fined tuned model.
    """
    # spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
    model.train() if training else model.eval()
    spt_embeddings_t = get_embedding(spt_x_t, model).detach()
    qry_embeddings_t = get_embedding(qry_x_t, model).detach()
    if target_type == 'classification':
        spt_embeddings_t = spt_embeddings_t.view(spt_embeddings_t.size(0), -1).cpu().numpy()
        qry_embeddings_t = qry_embeddings_t.view(qry_embeddings_t.size(0), -1).cpu().numpy()
        spt_y_t = spt_y_t.view(-1).cpu().numpy()
        qry_y_t = qry_y_t.view(-1).cpu().numpy()

        # Inner-Adapt final layer with spt set
        mdl = LogisticRegression(random_state=0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        mdl.fit(spt_embeddings_t, spt_y_t)

        # Predict using adapted final layer with qrt set
        if classifier == 'LR':
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                     multi_class='multinomial')
            clf.fit(spt_embeddings_t, spt_y_t)
            # query_y_pred_t = clf.predict(qry_embeddings_t)
            query_y_probs_t = clf.predict_proba(qry_embeddings_t)

            # - get layer_to_replace e.g. model.cls
            module: nn.Module = getattr_model(model, layer_to_replace)
            assert module is not None, f'Final layer module is None instead of a pytorch module see: {module=}'
            # assert module is model.model.cls

            # - replace weights into model
            # coef_ndarray of shape (1, n_features) or (n_classes, n_features)
            new_weights: np.ndarray = clf.coef_  # (n_classes, n_features) -> [n_features, n_classes]
            new_biases: np.ndarray = clf.intercept_  # (n_classes,) -> [n_features,]
            num_classes, num_features = new_weights.shape[0], new_weights.shape[1]  # in_features, out_features=Dout
            module.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(torch.float32))
            module.bias = torch.nn.Parameter(torch.from_numpy(new_biases).to(torch.float32))
            assert module.weight.size() == torch.Size([num_classes, num_features]), f'Error not the same: ' \
                                                                                    f'{module.weight.size()}, {torch.Size([num_features, num_classes])}'
            assert module.bias.size() == torch.Size([num_classes])
            # query_y_probs_t = clf.predict_proba(qry_embeddings_t[:2, :])
            # out_mdl = torch.softmax(mdl(torch.from_numpy(qry_embeddings_t[2:, :]), dim=1))
            # assert np.isclose(out_mdl.detach().cpu().numpy(), query_y_probs_t).all()
        elif classifier == 'NN':
            query_y_pred_t = NN(spt_embeddings_t, spt_y_t, qry_embeddings_t)
            raise ValueError(f'Not tested {classifier}')
        elif classifier == 'Cosine':
            query_y_pred_t = Cosine(spt_embeddings_t, spt_y_t, qry_embeddings_t)
            raise ValueError(f'Not tested {classifier}')
        # acc
        # qry_loss_t = log_loss(qry_y_t, query_y_probs_t)
        # qry_acc_t = metrics.accuracy_score(query_y_pred_t, qry_y_t)
    elif target_type == 'regression':
        # Inner-Adapt final adapted layer with spt set
        # mdl = LinearRegression().fit(spt_embeddings_t, spt_y_t)
        # # Predict using adapted final layer with qrt set
        # query_y_pred_t = torch.Tensor(mdl.predict(qry_embeddings_t))
        # qry_loss_t = args.criterion(query_y_pred_t, qry_y_t)
        # qry_acc_t = r2_score_from_torch(qry_y_t, query_y_pred_t)
        assert False
    else:
        raise ValueError(f'Not implement: {target_type}')
    return model


def get_embedding(x: Tensor, model: nn.Module) -> Tensor:
    """ apply f until the last layer, instead return that as the embedding """
    out = x
    # if it has a get embedding later
    if hasattr(model, 'get_embedding'):
        out = model.get_embedding(x)
        return out
    # for l2l
    if hasattr(model, 'features'):
        out = model.features(x)
        return out
    # for handling models with self.model.features self.model.cls format
    if hasattr(model, 'model'):
        if hasattr(model.model, 'features'):
            out = model.model.features(x)
            return out
    # # for hugging face (HF) models
    # from transformers import PreTrainedModel
    # if isinstance(model, PreTrainedModel):
    #     out = model.model(x)
    #     hidden_states = out.last_hidden_state
    #     # Get the CLS token's features (position 0)
    #     cls_features = hidden_states[:, 0]
    #     return out
    # for handling synthetic base models
    # https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
    for name, m in model.named_children():
        if 'final' in name:
            return out
        if 'l2' in name and 'final' not in name:  # cuz I forgot to write final...sorry!
            return out
        if name == 'fc':
            return out
        out = m(out)
    raise ValueError(
        'Your model does not have an explicit point where the embedding starts (i.e. the word final) or other bug in model')


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred


def Proto(support, support_ys, query, opt):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = - ((query - support) ** 2).sum(-1)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,))
    return pred


# - tests

def features_vit_pre_train():
    # - ViT feature extractor
    from transformers import ViTFeatureExtractor, ViTModel
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.argparse_uu.meta_learning import get_args_vit_mdl_maml_l2l_agent_default
    args: Namespace = get_args_vit_mdl_maml_l2l_agent_default()
    from uutils.torch_uu.distributed import set_devices
    set_devices(args)
    get_and_create_model_opt_scheduler_for_run(args)
    print(f'{type(args.model)=}')
    x = torch.randn(25, 3, 84, 84)
    features = get_embedding(x, args.model)
    print(f'{features.shape=}')
    print()

    # # - real data
    # from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import \
    #     forward_pass_with_pretrain_convergence_ffl_meta_learner
    # forward_pass_with_pretrain_convergence_ffl_meta_learner()




if __name__ == '__main__':
    print('__main__ started!')
    import time

    start = time.time()
    # test_f_rand_is_worse_than_f_avg()
    features_vit_pre_train()
    seconds = time.time() - start
    print(f'seconds = {seconds}, hours = {seconds / 60} \n\a')
