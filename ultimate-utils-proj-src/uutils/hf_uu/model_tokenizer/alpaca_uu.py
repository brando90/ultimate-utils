"""
As far as I understand the model and tokenizer have to match. So it makes sense to me to have helper code or my code
that returns both together.
"""

# -- from trainer.py

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
# )
#
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
#     model_max_length=training_args.model_max_length,
#     padding_side="right",
#     use_fast=False,
# )
# special_tokens_dict = dict()
# if tokenizer.pad_token is None:
#     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
# if tokenizer.eos_token is None:
#     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
# if tokenizer.bos_token is None:
#     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
# if tokenizer.unk_token is None:
#     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
#
# smart_tokenizer_and_embedding_resize(
#     special_tokens_dict=special_tokens_dict,
#     tokenizer=tokenizer,
#     model=model,
# )

#  -- from supervised.py (alpaca_farm)
# if training_args.deepspeed is not None:
#     ctx_mgr = contextlib.nullcontext()
#     device_map = None
#     low_cpu_mem_usage = None
# elif training_args.initialize_model_on_cpu:
#     ctx_mgr = contextlib.nullcontext()
#     device_map = None
#     low_cpu_mem_usage = True
# else:
#     ctx_mgr = common.staggered_object_creation(
#         local_rank=training_args.local_rank, world_size=training_args.world_size
#     )
#     device_map = {"": training_args.device.index}
#     low_cpu_mem_usage = True
#
# with ctx_mgr:
#     model: transformers.PreTrainedModel = common.make_generative_lm(
#         model_name_or_path=model_args.model_name_or_path,
#         flash_attn=training_args.flash_attn,
#         fp16=training_args.fp16,
#         bf16=training_args.bf16,
#         config=transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
#         cache_dir=training_args.cache_dir,
#         low_cpu_mem_usage=low_cpu_mem_usage,
#         device_map=device_map,
#     )
#     common.let_model_save_mem_when_zero_grad(model)
#
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
#     model_max_length=training_args.model_max_length,
#     padding_side="right",  # Ensures properly masking out the source tokens.
#     use_fast=training_args.use_fast_tokenizer,
# )
# tokenizer.padding = training_args.padding
#
# # Collect special tokens. Only add if non-existent.
# special_tokens_dict = dict(additional_special_tokens=[])
# if tokenizer.pad_token is None:
#     special_tokens_dict["pad_token"] = training_args.pad_token
# if tokenizer.eos_token is None:
#     special_tokens_dict["eos_token"] = constants.DEFAULT_EOS_TOKEN
# if tokenizer.bos_token is None:
#     special_tokens_dict["bos_token"] = constants.DEFAULT_BOS_TOKEN
# if tokenizer.unk_token is None:
#     special_tokens_dict["unk_token"] = constants.DEFAULT_UNK_TOKEN
# utils.stable_resize_token_embeddings_and_tokenizer(model, tokenizer, special_tokens_dict)
