"""
Goal: compute the diversity using task2vec as the distance between tasks.

Idea: diversity should measure how different **two tasks** are. e.g. the idea could be the distance between distributions
e.g. hellinger, total variation, etc. But in real tasks high dimensionality of images makes it impossible to compute the
integral needed to do this properly. One way is to avoid integrals and use metrics on parameters of tasks -- assuming
a closed for is known for them & the parameters of tasks is known.
For real tasks both are impossible (i.e. the generating distribution for a n-way, k-shot task is unknown and even if it
were known, there is no guarantee a closed form exists).
But FIM estimates something like this. At least in the classical MLE case (linear?), we have
    sqrt(n)(theta^MLE(n) - theta^*) -> N(0, FIM^-1) wrt P_theta*, in distribution and n->infinity
so high FIM approximately means your current params are good.
But I think the authors for task2vec are thinking of the specific FIM (big matrix) as the specific signature for the
task. But when is this matrix (or a function of it) a good representation (embedding) for the task?
Or when is it comparable with other values for the FIM? Well, my guess is that if they are the same arch then the values
between different FIM become (more) comparable.
If the weights are the same twice, then the FIM will always be the same. So I think the authors just meant same network
as same architecture. Otherwise, if we also restrict the weights the FIM would always be the same.
So given two FIM -- if it was in 1D -- then perhaps you'd choose the network with the higher FI(M).
If it's higher than 1D, then you need to consider (all or a function of) the FIM.
The distance between FIM (or something like that seems good) given same arch but different weights.
They fix final layer but I'm not sure if that is really needed.
Make sure to have lot's of data (for us a high support set, what we use to fine tune the model to get FIM for the task).
We use the entire FIM since we can't just say "the FIM is large here" since it's multidimensional.
Is it something like this:
    - a fixed task with a fixed network that predicts well on it has a specific shape/signature of the FIM
    (especially for something really close, e.g. the perfect one would be zero vector)
    - if you have two models, then the more you change the weights the more the FIM changes due to the weights being different, instead of because of the task
    - minimize the source of the change due to the weights but maximize it being due to the **task**
    - so change the fewest amount of weights possible?


Paper says embedding of task wrt fixed network & data set corresponding to a task as:
    task2vec(f, D)
    - fixed network and fixed feature extractor weights
    - train final layer wrt to current task D
    - compute FIM = FIM(f, D)
    - diag = diag(FIM)  # ignore different filter correlations
    - task_emb = aggregate_same_filter_fim_values(FIM)  # average weights in the same filter (since they are usually depedent)

div(f, B) = E_{tau1, tau2} E_{spt_tau1, spt_tau2}[d(task2vec(f, spt_tau1),task2vec(f, spt_tau2)) ]

for div operator on set of distances btw tasks use:
    - expectation
    - symmetric R^2 (later)
    - NED (later)

----
l2l comments:
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    task_dataset: TaskDataset = task_dataset,  # eg args.tasksets.train

    BenchmarkTasksets = contains the 3 splits which have the tasks for each split.
    e.g. the train split is its own set of "tasks_dataset.train = {task_i}_i"

"""
from typing import Optional

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import learn2learn
import numpy as np
import torch.utils.data
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

import diversity_src.diversity.task2vec_based_metrics.task2vec as task2vec
import diversity_src.diversity.task2vec_based_metrics.task_similarity as task_similarity
# from dataset import TaskDataset
# from models import get_model
# from task2vec import Embedding, Task2Vec, ProbeNetwork
from diversity_src.diversity.task2vec_based_metrics.models import get_model
from diversity_src.diversity.task2vec_based_metrics.task2vec import Embedding, Task2Vec, ProbeNetwork


def mds_loader_to_list_tensor_loader(episodic_mds_loader: DataLoader, num_tasks_to_consider: int) -> list[Tensor]:
    """
    This needs to receive an mds data loader & the number of tasks to consider for it's meta-batch (i.e. it's batch
    of tasks).

    Instead of [B, n_b*k_b, C, H, W] -> B*[n_b*k_n, C, H, W] and skip the issue of concatenating tensors of veriable sizes (by some padding?).
    """
    meta_batch: list[Tensor] = []
    for _ in range(num_tasks_to_consider):
        batch: Tensor = next(iter(episodic_mds_loader))
        assert len(batch.size()) == 5
        # spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
        spt_x, spt_y, qry_x, qry_y = batch
        meta_batch.append((spt_x, spt_y, qry_x, qry_y))
    yield meta_batch


# - probe network code

def get_5CNN_random_probe_network() -> ProbeNetwork:
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner
    probe_network: nn.Module = get_default_learner()
    # just going to force to give it the fields and see if it works
    probe_network.classifier = probe_network.cls
    return probe_network


def get_probe_network_from_ckpt():
    # TODO
    pass


# - diversity code

def split_task_spt_qrt_points(task_data, device, shots, ways):
    # [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
    data, labels = task_data
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    # [n*(k+k_eval), C, H, W] -> [n*k, C, H, W] and [n*k_eval, C, H, W]
    (support_data, support_labels), (query_data, query_labels) = learn2learn.data.partition_task(
        data=data,
        labels=labels,
        shots=shots,  # shots to separate to two data sets of size shots and k_eval
    )
    # checks coordinate 0 of size() [n*(k + k_eval), C, H, W]
    assert support_data.size(0) == shots * ways, f' Expected {shots * ways} but got {support_data.size(0)}'
    # checks [n*k] since these are the labels
    assert support_labels.size() == torch.Size([shots * ways])

    return (support_data, support_labels), (query_data, query_labels)


class FSLTaskDataSet(Dataset):

    def __init__(self,
                 spt_x: Tensor,
                 spt_y: Tensor,
                 qry_x: Tensor,
                 qry_y: Tensor,

                 few_shot_split: str = 'qry',
                 ):
        """
        Note:
            - size of tensors are [M, C, H, W] but remember it comes from a batch of
            tasks of size [B, M, C, H, W]
        """
        self.spt_x, self.spt_y, self.qry_x, self.qry_y = spt_x, spt_y, qry_x, qry_y
        self.few_shot_split = few_shot_split
        # - weird hack, since task2vec code get_loader does labels = list(trainset.tensors[1].cpu().numpy())
        if self.few_shot_split == 'spt':
            self.tensors = (spt_x, spt_y)
        elif self.few_shot_split == 'qry':
            self.tensors = (qry_x, qry_y)
        else:
            raise ValueError(f'Error needs to be spt or qrt but got {self.few_shot_split}')

    def __len__(self):
        # TODO
        if self.few_shot_split == 'spt':
            return self.spt_x.size(0)  # not 1 since we gave it a single task
        elif self.few_shot_split == 'qry':
            return self.qry_x.size(0)  # not 1 since we gave it a single task
        else:
            raise ValueError(f'Error needs to be spt or qrt but got {self.few_shot_split}')

    def __getitem__(self, idx: int):
        """
        typical implementation:

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        """
        # TODO
        if self.few_shot_split == 'spt':
            return self.spt_x[idx], self.spt_y[idx]
        elif self.few_shot_split == 'qry':
            return self.qry_x[idx], self.qry_y[idx]
        else:
            raise ValueError(f'Error needs to be spt or qrt but got {self.few_shot_split}')


def get_task_embeddings_from_few_shot_l2l_benchmark(tasksets: BenchmarkTasksets,
                                                    probe_network: ProbeNetwork,
                                                    num_tasks_to_consider: int,

                                                    split: str = 'validation',
                                                    split_task: bool = False,
                                                    classifier_opts: Optional = None,
                                                    ) -> list[task2vec.Embedding]:
    """

    note:
        - you can't just pass a nn.Module, it has to have the classifier setter & getter due to how task2vec works. My
        guess is that since it does have to fine tune the final layer before computing FIM, then it requires to have
        access to the modules considered the final layer.
        - note that the task2vec code has side effects on your probe network, so you have to have some type of code that
        "removes" those side effects. Two options is if the function takes in a probe_network directly then it has to
        create a deep copy when feeding it to Task2Vec. Otherwise another option is to load a new probe nework every
        time we create a new fsl.
    """
    # - get the data set of (n-way, k-shot) tasks
    from learn2learn.data import TaskDataset
    task_dataset: TaskDataset = getattr(tasksets, split)  # tasksets.train

    # - compute embeddings for tasks
    embeddings: list[task2vec.Embedding] = []
    for task_num in range(num_tasks_to_consider):
        print(f'\n--> {task_num=}\n')
        # - Samples all data data for spt & qry sets for current task: thus size [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
        task_data: list = task_dataset.sample()  # data, labels
        if split_task:
            # - split to spt & qry sets
            raise ValueError('Not implemented to split task data set using sqt & qry ala l2l.')
            # split_task_spt_qrt_points(task_data, device, shots, ways)
        else:
            # - use all the data in the task
            data, labels = task_data
            fsl_task_dataset: Dataset = FSLTaskDataSet(spt_x=None, spt_y=None, qry_x=data, qry_y=labels)
            print(f'{len(fsl_task_dataset)=}')
            data: Tensor = data
            # embedding: task2vec.Embedding = Task2Vec(deepcopy(probe_network)).embed(fsl_task_dataset)
            embedding: task2vec.Embedding = Task2Vec(deepcopy(probe_network), classifier_opts=classifier_opts).embed(
                fsl_task_dataset)
            print(f'{embedding.hessian.shape=}')
        embeddings.append(embedding)
    return embeddings


def get_task_embeddings_from_few_shot_dataloader(args: Namespace,
                                                 dataloaders: dict,
                                                 probe_network: ProbeNetwork,
                                                 num_tasks_to_consider: int,
                                                 split: str = 'validation',
                                                 classifier_opts: Optional = None,
                                                 ) -> list[task2vec.Embedding]:
    """
    Returns list of task2vec embeddings using the normal pytorch dataloader interface.
    Should work for torchmeta data sets & meta-data set (MDS).

    Algorithm:
    - sample the 4 tuples of T tasks
    - loop through each task & use it as data to produce the task2vec
    """
    # - get the data set of (n-way, k-shot) tasks
    # loader = args.dataloader[split]
    loader = dataloaders[split]

    # -
    from uutils.torch_uu import process_meta_batch
    batch = next(iter(loader))  # batch [B, n*k, C, H, W] or B*[n_b*k_b, C, H, W]
    spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

    # - compute embeddings for tasks
    embeddings: list[task2vec.Embedding] = []
    for t in range(num_tasks_to_consider):
        print(f'\n--> task_num={t}\n')
        spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]

        # concatenate the support and query sets to get the full task's data and labels
        data = torch.cat((spt_x_t, qry_x_t),0)
        labels = torch.cat((spt_y_t, qry_y_t),0)

        #print(data.shape, labels.shape)
        fsl_task_dataset: Dataset = FSLTaskDataSet(spt_x=None, spt_y=None, qry_x=data, qry_y=labels)

        print(f'{len(fsl_task_dataset)=}')
        embedding: task2vec.Embedding = Task2Vec(deepcopy(probe_network), classifier_opts=classifier_opts).embed(
            fsl_task_dataset)
        print(f'{embedding.hessian.shape=}')
        embeddings.append(embedding)
    return embeddings


def compute_diversity(distance_matrix: np.array,
                      remove_diagonal: bool = True,
                      variance_type: str = 'ci_0.95',
                      ) -> tuple[float, float]:
    """
    Computes diversity using task2vec embeddings from a distance matrix as:
        div(B, f) = E_{t1, t2 ~ p(t|B)} E_{X1 ~ p(X|t1) X1 ~ p(X|t2)}[dist(emb(X1, f), emb(X2, f)]
    """
    div, ci = task_similarity.stats_of_distance_matrix(distance_matrix, remove_diagonal, variance_type)
    return div, ci


# - tests

def plot_distance_matrix_and_div_for_MI_test():
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    import uutils
    from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    from uutils.argparse_uu.meta_learning import parse_args_meta_learning
    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility

    # - get args for test
    args: Namespace = parse_args_meta_learning()
    args.batch_size = 2
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'
    args = fix_for_backwards_compatibility(args)  # TODO fix me
    uutils.print_args(args)

    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)

    # - create probe_network
    # probe_network: nn.Module = get_default_learner()
    # probe_network: ProbeNetwork = get_5CNN_random_probe_network()
    # probe_network: ProbeNetwork = get_model('resnet34', pretrained=True, num_classes=5)
    probe_network: ProbeNetwork = get_model('resnet18', pretrained=True, num_classes=5)

    # - compute task embeddings according to task2vec
    print(f'number of tasks to consider: {args.batch_size=}')
    embeddings: list[Tensor] = get_task_embeddings_from_few_shot_l2l_benchmark(args.tasksets,
                                                                               probe_network,
                                                                               num_tasks_to_consider=args.batch_size)
    print(f'\n {len(embeddings)=}')

    # - compute distance matrix & task2vec based diversity
    # to demo task2vec, this code computes pair-wise distance between task embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')

    # this code is similar to above but put computes the distance matrix internally & then displays it
    task_similarity.plot_distance_matrix(embeddings, labels=list(range(len(embeddings))), distance='cosine')

    div, ci = task_similarity.stats_of_distance_matrix(distance_matrix)
    print(f'Diversity: {(div, ci)=}')


def mds_loop():
    pass


if __name__ == '__main__':
    # plot_distance_matrix_and_div_for_MI_test()
    print('Done! successful!\n')
