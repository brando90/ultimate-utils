
# Difference between src_mask and src_key_padding_mask

The general thing is to notice the difference between the use of the tensors `_mask` vs `_key_padding_mask`.
Inside the transformer when attention is done we usually get an squared intermediate tensor with all the comparisons
of size `[Tx, Tx]` (for the input to the encoder), `[Ty, Ty]` (for the shifted output - one of the inputs to the decoder)
and `[Ty, Tx]` (for the memory mask - the attention between output of encoder/memory and input to decoder/shifted output).
 
So we get that this are the uses for each of the masks in the transformer
(note the notation from the pytorch docs is as follows where `Tx=S is the source sequence length` 
(e.g. max of input batches), 
`Ty=T is the target sequence length` (e.g. max of target length), 
`B=N is the batch size`, 
`D=E is the feature number`):

1. src_mask `[Tx, Tx] = [S, S]` – the additive mask for the src sequence (optional).
   This is applied when doing `atten_src + src_mask`. I'm not sure of an example input - see tgt_mask for an example
   but the typical use is to add `-inf` so one could mask the src_attention that way if desired.
   If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
   If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. 
   If a FloatTensor is provided, it will be added to the attention weight.

2. tgt_mask `[Ty, Ty] = [T, T]` – the additive mask for the tgt sequence (optional).
   This is applied when doing `atten_tgt + tgt_mask`. An example use is the diagonal to avoid the decoder from cheating. 
   So the tgt is right shifted, the first tokens are start of sequence token embedding SOS/BOS and thus the first 
   entry is zero while the remaining. See concrete example at the appendix.
   If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
   If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. 
   If a FloatTensor is provided, it will be added to the attention weight.

3. memory_mask `[Ty, Tx] = [T, S]`– the additive mask for the encoder output (optional).
   This is applied when doing `atten_memory + memory_mask`.
   Not sure of an example use but as previously, adding `-inf` sets some of the attention weight to zero.
   If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
   If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. 
   If a FloatTensor is provided, it will be added to the attention weight. 

4. src_key_padding_mask `[B, Tx] = [N, S]` – the ByteTensor mask for src keys per batch (optional).
   Since your src usually has different lengths sequences it's common to remove the padding vectors 
   you appended at the end.
   For this you specify the length of each sequence per example in your batch.
   See concrete example in appendix. 
   If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
   If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. 
   If a FloatTensor is provided, it will be added to the attention weight.

5. tgt_key_padding_mask `[B, Ty] = [N, t]` – the ByteTensor mask for tgt keys per batch (optional). 
   Same as previous. 
   See concrete example in appendix.
   If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
   If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. 
   If a FloatTensor is provided, it will be added to the attention weight.

6. memory_key_padding_mask `[B, Tx] = [N, S]` – the ByteTensor mask for memory keys per batch (optional).
   Same as previous.
   See concrete example in appendix.
   If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
   If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. 
   If a FloatTensor is provided, it will be added to the attention weight.
   
# Appendix

Examples from pytorch tutorial (https://pytorch.org/tutorials/beginner/translation_transformer.html):

## 1 src_mask example

```
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
```

returns a tensor of booleans of size `[Tx, Tx]`:
```
tensor([[False, False, False,  ..., False, False, False],
         ...,
        [False, False, False,  ..., False, False, False]])
```

# 2 tgt_mask example

```
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1)
    mask = mask.transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
```

generates the diagonal for the right shifted output which the input to the decoder.

```
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
         -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
         -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
         -inf, -inf, -inf],
         ...,
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
```

usually the right shifted output has the BOS/SOS at the beginning and it's the tutorial gets the right shift simply
by appending that BOS/SOS at the front and then triming the last element with `tgt_input = tgt[:-1, :]`.

# 3 _padding

The padding is just to mask the padding at the end. 
The src padding is usually the same as the memory padding.
The tgt has it's own sequences and thus it's own padding.
Example:
```
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    memory_padding_mask = src_padding_mask
```

Output:

```
tensor([[False, False, False,  ...,  True,  True,  True],
        ...,
        [False, False, False,  ...,  True,  True,  True]])
```

----

The answers are sort of spread around but I found only these 3 references being useful 
(the seperate layers stuff wasn't very useful honesty): 
 - long tutorial: https://pytorch.org/tutorials/beginner/translation_transformer.html
 - MHA docs: https://pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention
 - transformer docs: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html 
 - This post: https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask/68396781#68396781