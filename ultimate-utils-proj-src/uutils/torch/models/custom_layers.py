import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F

import uutils

import math

import time
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Transformer, TransformerDecoder

import uutils

class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class PositionalEncoding(nn.Module):
    """
    Inspired from transformer tutorial: https://pytorch.org/tutorials/beginner/translation_transformer.html
    Note: there are no learnable parameters. The only layer inside of this is a dropout.

    pos = token location
    i = d = dimension of embedding of token
    PE[pos, i] =
        sin(pos/10_000^2i/D) if i = 2k (even)
        cos(pos/10_000^2i/D) if i = 2k+1 (odd)
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.1,  # same default as pytorch's transformer layer
                 maxlen: int = 5_000,
                 batch_first:bool = False,
                 ):
        """

        :param embed_dim:
        :param dropout:
        :param maxlen: max length of the pos embedding. This usually seems too large but note that
        pos encodings (usually) don't have batch dimension - so this is a single tensor
        that we can just trim as needed according to the real max length in a batch.
        :param batch_first:
        """
        super().__init__()
        if not batch_first:
            logging.log(logging.WARN, f"Warning: Brando usually likes batch first but its not true: {batch_first=}")
        self.batch_first = batch_first
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        # todo - improve comment
        den = torch.exp(- torch.arange(0, embed_dim, 2)* math.log(10_000) / embed_dim)
        # position
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # create empty positional embedding to fill up
        pos_embedding = torch.zeros((maxlen, embed_dim))
        # 0::2 == start at zero and get every 2 (implements get even)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # 1::2 == start at 1 and get every 2 (implements get odd)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # places a singleton 1 dimension where the batch goes for proper broadcasting
        if batch_first:
            pos_embedding = pos_embedding.unsqueeze(0)
        else:
            pos_embedding = pos_embedding.unsqueeze(1)
            #pos_embedding = pos_embedding.unsqueeze(-2)
        assert pos_embedding.size() == torch.Size([1, self.maxlen, self.embed_dim]), f'This will not broadcast correct with the input tokens because we have ' \
                                                                                     f'size: {pos_embedding.size()}'
        self.register_buffer('pos_embedding', pos_embedding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_seq_embedding: Tensor) -> Tensor:
        """
        Computes positional embeddingi for the given batch of sequence token embeddings.
        Important: make sure you know if your model's shape has batch_first or not to have this layer work
        properly.
        Note we always assume last dimension is the dimension of the embeddings

        :param token_embedding:
        :return:
        """
        assert len(token_seq_embedding.size()) == 3, f'Expected a token of shape B,T,D but got: {token_seq_embedding.size()=}'
        # if its batch first then the length of the sequence the second element, last element is always embed_dim
        if self.batch_first:
            # B, T, D = token_seq_embedding.size()
            T = token_seq_embedding.size(1)
            pos_enc = self.pos_embedding[:, :T, :]
            assert pos_enc.size() == torch.Size([1, T, self.embed_dim])
        else:
            # T, B, D = token_seq_embedding.size()
            T = token_seq_embedding.size(0)
            pos_enc = self.pos_embedding[:T, :, :]
            assert pos_enc.size() == torch.Size([T, 1, self.embed_dim])
        # get the number of positional embedding relevant for this batch of sequences
        out = self.dropout(token_seq_embedding + pos_enc)
        return out

class ResNet(torch.nn.Module):
    """https://stackoverflow.com/questions/57229054/how-to-implement-my-own-resnet-with-torch-nn-sequential-in-pytorch"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

# -- FNN modules --

# class FNN1(nn.Module):
#     def __init__(self, in_dim, embed_dim, num_tactic_hashes):
#         super().__init__()
#         self.module = nn.Sequential(OrderedDict([
#             ('fc0', nn.Linear(in_features=in_dim, out_features=embed_dim)),
#             ('SELU0', nn.SELU()),
#             ('fc1', nn.Linear(in_features=embed_dim, out_features=num_tactic_hashes))
#         ]))
#
#     def forward(self, x):
#         return self.module(x)

# -- Classifier modules --

class Cls1(nn.Module):
    def __init__(self, in_dim, embed_dim, out_features):
        super().__init__()
        self.fnn = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=in_dim, out_features=in_dim)),
            ('SELU0', nn.SELU()),
            ('fc1', nn.Linear(in_features=in_dim, out_features=in_dim)),
        ]))
        self.norm = nn.SELU()
        self.lin_output = nn.Linear(in_features=in_dim, out_features=out_features)

    def forward(self, x):
        # block 1 TBF
        x = self.norm(x + self.fnn(x))
        # cls
        out = self.lin_output(x)
        return out

class Cls2(nn.Module):
    def __init__(self, in_dim, embed_dim, out_features):
        super().__init__()
        self.fnn = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=in_dim, out_features=in_dim)),
            ('ReLU0', nn.ReLU()),
            ('fc1', nn.Linear(in_features=in_dim, out_features=in_dim)),
        ]))
        self.norm = nn.LayerNorm(normalized_shape=in_dim)
        self.lin_output = nn.Linear(in_features=in_dim, out_features=out_features)

    def forward(self, x):
        # block 1 TBF
        x = self.norm(x + self.fnn(x))
        # cls
        out = self.lin_output(x)
        return out

# -- Blocks tried --

class ResNet1(nn.Module):
    def __init__(self, in_dim, embed_dim, out_features):
        super().__init__()
        self.fnn = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=in_dim, out_features=embed_dim)),
            ('SELU0', nn.SELU()),
            ('fc1', nn.Linear(in_features=embed_dim, out_features=out_features)),
        ]))
        self.norm = nn.SELU()

    def forward(self, x):
        # TBF
        out = self.norm(x + self.fnn(x))
        return out

class ResNet2(nn.Module):
    def __init__(self, in_dim, embed_dim, out_features):
        super().__init__()
        self.fnn = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=in_dim, out_features=embed_dim)),
            ('ReLU0', nn.ReLU()),
            ('fc1', nn.Linear(in_features=embed_dim, out_features=out_features)),
        ]))
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        # TBF
        out = self.norm(x + self.fnn(x))
        return out

class TransformerTreeBlock(nn.Module):
    #def __init__(self, in_dim, embed_dim, out_features):
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=1028):
        super().__init__()
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

    def forward(self, x: torch.Tensor, y: torch.Tensor, batch_first=False) -> torch.Tensor:
        """
        [B, Tx, D] -> [B, D]
        [Tx, B, D] -> [B, D]

        Intended use:
            [1, Tx+Ty, D] x [1, Tx+Ty, D] -> [1, Ty, D]
        :return:
        """
        # - if its a single sequence
        if len(x.size()) == 2:
            out = self.forward_single_sequence(x, y)
            return out
        # - if its a batch of sequences
        if batch_first:
            x = x.view(x.size(1), x.size(0), -1)  # [B, Tx, D] -> [Tx, B, D]
        out = self.transformer(src=x, tgt=y)
        return out

    def forward_single_sequence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        [Tx, D] x [Ty, D] -> [1, Ty, D]

        [Tx, D] x [D] -> [1, Tx, D] x [1, 1, D] -> [1, Ty, D]
        [Tx, D] x [Ty, D] -> [1, Tx, D] x [1, Ty, D] -> [1, Ty, D]
        """
        Tx, D = x.size(0), x.size(1)
        if len(y.size()) == 1:
            # [Tx, D] x [D] -> [1, Tx, D] x [1, 1, D]
            Ty, D = 1, y.size(0)
        else:
            # [Tx, D] x [Ty, D] -> [1, Tx, D] x [1, Ty, D]
            Ty, D = y.size(0), y.size(1)
        # reshape/review tensors
        x = x.view(1, Tx, D)
        y = y.view(1, Ty, D)
        # batch_first so that it eliminates the Tx dimension [1, Tx, D] -> [1, 1, D]
        set_embedding = self.forward(x, y, batch_first=True)
        return set_embedding.unsqueeze(1)  # [1, 1, D] --> [1, D]

# -- Block for Batched TreeNN --

class ResNetBatchCons(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x, cons_one_hot):
        """
        The is that we use the 1-hot vector from cons to select the right weights for FNN.


        Idea:
        - we want the 0-1 vector from the cons to select the FNN to use.
            - do that with a normal matrix multiplication
            - but encode the entire weights as a flatten weights of size D_embd * D_out
            - then do a conv2d with the input (the filter is the embedding of the input word)

        For one (batched/vectorized) node computation (for 1 hidden layer case):

            out = (x_cons W^(l1) * e_word ) ] (x_cons W^(l2) * x_prev_layer

        - where x_cons is torch.Tensor([batch, D_cons])
        - where e_word is torch.Tensor([batch, D_emb])
        - where ] indicates a ReLU/activation
        - where * is a convolution
        - l1, l2 are layers

        note the general recursive one is (without norm or skip connections):

            out^<li> = Activation(x_cons W^(l1) * out^<l_i-1 )

        Note that the above is simplified and it's missing the residual connections and norm layers.
        - put the norm after the FC
        - there is different ways to do skip connections (depending what a block means for us).
            - transformers do it after self attention or after an entire FNN
            - so one way is to do it after each step/activation or at the very end of the forward path

        :param x: [B, embed_dim]
        :param cons_one_hot: [B, cons_dim] cons_dim ~ 21
        :return:
        """
        pass

# -- self mean layers

class SelfMean(nn.Module):

    def __init__(self, d_model=256, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=1028):
        """

        Note dim_feedforwards satisfies 512*4 = 2048.
        If d_model=256 then 256*4 = 1024
        It might be useful to keep that.


        :param d_model:
        :param nhead:
        :param num_encoder_layers:
        :param num_decoder_layers:
        :param dim_feedforward:
        """
        super().__init__()
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)


    def forward(self, x: torch.Tensor, batch_first=False) -> torch.Tensor:
        """
        Does a mean over a variable length sequence where the weights are learnable.
        A replacement for x.mean() layers.
        Motivation: there are times we have tensors of variable length and we'd like to learn a linear combination
        over it's values (e.g. a sequence of tokens). But since it's of variable length one cannot use a fixed matrix.
        Thus, one can use a simple version of attention:
            e_self_mean[b] = sum^{Tx}_{i} e_i alpha(e[b]_i, e.mean())
        no we have a variable length learnable average.

        [B, Tx, D] -> [B, D]
        [Tx, B, D] -> [B, D]

        :return:
        """
        if batch_first:
            x = x.view(x.size(1), x.size(0), -1)  # [B, Tx, D] -> [Tx, B, D]
        # usually means mean over squence Tx
        x_mean = x.mean(dim=0, keepdim=True)  # other options exist of course
        x_self_mean = self.transformer(src=x, tgt=x_mean)
        return x_self_mean.squeeze(0)

    def forward_single_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[Tx, D] -> [1, D]"""
        Tx, D = x.size(0), x.size(1)
        set_embeddings = x.view(1, Tx, D)
        # batch_first so that it eliminates the Tx dimension [1, Tx, D] -> [1, 1, D]
        set_embedding = self.forward(set_embeddings, batch_first=True)
        return set_embedding

class SelfMeanSimple(SelfMean):

    def __init__(self, d_model=256, nhead=4, num_decoder_layers=1, dim_feedforward=1028):
        """

        Note dim_feedforwards satisfies 512*4 = 2048.
        If d_model=256 then 256*4 = 1024
        It might be useful to keep that.


        :param d_model:
        :param nhead:
        :param num_encoder_layers:
        :param num_decoder_layers:
        :param dim_feedforward:
        """
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, x: torch.Tensor, batch_first=False) -> torch.Tensor:
        """
        Does a mean over a variable length sequence where the weights are learnable.
        A replacement for x.mean() layers.
        Motivation: there are times we have tensors of variable length and we'd like to learn a linear combination
        over it's values (e.g. a sequence of tokens). But since it's of variable length one cannot use a fixed matrix.
        Thus, one can use a simple version of attention:
            e_self_mean[b] = sum^{Tx}_{i} e_i alpha(e[b]_i, e.mean())
        no we have a variable length learnable average.

        [B, Tx, D] -> [B, D]
        [Tx, B, D] -> [B, D]

        :return:
        """
        if batch_first:
            x = x.view(x.size(1), x.size(0), -1)  # [B, Tx, D] -> [Tx, B, D]
        # usually means mean over squence Tx
        x_mean = x.mean(dim=0, keepdim=True)  # other options exist of course
        x_self_mean = self.transformer(tgt=x_mean, memory=x)
        return x_self_mean.squeeze(0)

    def forward_single_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[Tx, D] -> [1, D]"""
        Tx, D = x.size(0), x.size(1)
        x = x.view(1, Tx, D)
        # batch_first so that it eliminates the Tx dimension [1, Tx, D] -> [1, 1, D]
        x = self.forward(x, batch_first=True)
        return x

# - tests

def test_self_mean_trans():
    batch_size = 4; Tx = 5; D = 256
    x = torch.randn(batch_size, Tx, D)
    mdl = SelfMean()
    x_bar = mdl(x, batch_first=True)
    print(x_bar.size())
    assert(x_bar.size() == torch.Size([batch_size, D]))
    mdl = SelfMeanSimple()
    x_bar = mdl(x, batch_first=True)
    print(x_bar.size())
    assert(x_bar.size() == torch.Size([batch_size, D]))
    #
    Tx = 5; D = 256
    x = torch.randn(Tx, D)
    mdl = SelfMean()
    x_bar = mdl.forward_single_sequence(x)
    print(x_bar.size())
    assert(x_bar.size() == torch.Size([1, D]))
    mdl = SelfMeanSimple()
    x_bar = mdl.forward_single_sequence(x)
    print(x_bar.size())
    assert(x_bar.size() == torch.Size([1, D]))

def test_batch_cons():
    pass

def test_positional_embedding_test():
    B, T, D = 2, 3, 4
    batch_src = torch.randn((B, T, D))

    pos_enc_layer = PositionalEncoding(D, batch_first=True)
    src = pos_enc_layer(batch_src)
    print(src.size())

if __name__ == '__main__':
    start = time.time()
    # test_batch_cons()
    # test_self_mean_trans()
    test_positional_embedding_test()
    print(f'{uutils.report_times(start)}')
