import torch
from torch import FloatTensor, nn, Tensor


def accuracy(output: Tensor,
             target: Tensor,
             topk: tuple[int] = (1,),
             reduction: str = 'mean'
             ) -> tuple[FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    if reduction is
        - "none" computes the 1 or 0 topk acc for each example so each entry is a tensor of size [B]
        - "mean" (default) compute the usual topk acc where each entry is acck of size [], single value tensor

    ref:
    - https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracies [top1st, top2nd, ...] depending on your topk input. Size [] or [B] depending on
        reduction type.
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
                y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # accuracy for the current topk for the whole batch,  [k, B] -> [B]
            indicator_which_topk_matched_truth = ind_which_topk_matched_truth.float().sum(dim=0)
            assert indicator_which_topk_matched_truth.size() == torch.Size([batch_size])
            # put a 1 in the location of the topk we allow if we got it right, only 1 of the k for each B can be 1.
            # Important: you can only have 1 right in the k dimension since the label will only have 1 label and our
            if reduction == 'none':
                topk_acc = indicator_which_topk_matched_truth
                assert topk_acc.size() == torch.Size([batch_size])
            elif reduction == 'mean':
                # compute topk accuracies - the model's ability to get it right within it's top k guesses/preds
                topk_acc = indicator_which_topk_matched_truth.mean()  # topk accuracy for entire batch
                assert topk_acc.size() == torch.Size([])
            else:
                raise ValueError(f'Invalid reduction type, got: {reduction=}')
            list_topk_accs.append(topk_acc)
        return tuple(list_topk_accs)  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def acc_test():
    B = 10000
    Dx, Dy = 2, 10
    mdl = nn.Linear(Dx, Dy)
    x = torch.randn(B, Dx)
    y_logits = mdl(x)
    y = torch.randint(high=Dy, size=(B,))
    print(y.size())
    acc1, acc5 = accuracy(output=y_logits, target=y, topk=(1, 5))
    print(f'{acc1=}')
    print(f'{acc5=}')

    accs1, accs5 = accuracy(output=y_logits, target=y, topk=(1, 5), reduction='none')
    print(f'{accs1=}')
    print(f'{accs5=}')
    print(f'{accs1.mean()=}')
    print(f'{accs5.mean()=}')
    print(f'{accs1.std()=}')
    print(f'{accs5.std()=}')
    print(f'{torch_compute_confidence_interval_classification_torch(accs1)=}')
    print(f'{torch_compute_confidence_interval_classification_torch(accs5)=}')

def prob_of_truth_being_inside_when_using_ci_as_std():
    """

    :return:
    """

    from scipy.integrate import quad
    # integration between x1 and x1
    def normal_distribution_function(x):
        import scipy.stats
        value = scipy.stats.norm.pdf(x, mean, std)
        return value
    mean, std = 0.0, 1.0

    x1 = mean - std
    x2 = mean + std

    res, err = quad(func=normal_distribution_function, a=x1, b=x2)

    print('Normal Distribution (mean,std):', mean, std)
    print('Integration bewteen {} and {} --> '.format(x1, x2), res)


if __name__ == '__main__':
    # acc_test()
    prob_of_truth_being_inside_when_using_ci_as_std()
    print('Done, success! \a')
