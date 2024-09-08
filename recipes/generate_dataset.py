# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from torch.distributed import destroy_process_group, init_process_group

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from functools import partial

from tqdm import tqdm


from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message
from torch.utils.data import DataLoader, DistributedSampler
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset


logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)
        
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0
        self._cfg = cfg


        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
            enable_kv_cache=cfg.enable_kv_cache,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._train_sampler, self._train_dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=False, # Force shuffle to False for generation
            batch_size=1,  # Force batch size to 1 for generation
        )

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()
        print(f"World size: {world_size}, Rank: {rank}")

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=-100, # the hard-coded classic 
            )
            if not packed
            else None,
        )

        if self._is_rank_zero:
            logger.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        enable_kv_cache: bool = True,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        if enable_kv_cache:
            with self._device:
                model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]

    @torch.no_grad()
    def generate(self, cfg: DictConfig):

        _has_compiled    = False
        world_size, rank = utils.get_world_size_and_rank()

        # worker buffer to store the generated tokens       
        rank_results = []
        pbar         = None 

        custom_generate_next_token = None  
        for batch_idx, batch in enumerate(self._train_dataloader):
            
            # NOTE: For debugging purposes, we can limit the number of batches
            #       processed by the recipe. This is useful for testing the
            #       recipe on a small subset of the data or for inspecting throughput.
            if batch_idx > cfg.get("limit_batches", float("inf")):
                break

            tokens, labels  = batch["tokens"], batch["labels"]
            is_prompt_mask  = labels == -100
            prompt_tokens   = tokens[is_prompt_mask]
            prompt_tokens   = prompt_tokens.to(device=self._device)

            # NOTE: Yes - compiling inside the loader makes for very ugly code
            ###########################  COMPILATION  ###############################
            if not _has_compiled and self._cfg.get("compile", True):
                if self._is_rank_zero:
                    logger.info("Starting compilation to improve generation performance ...")
                
                custom_generate_next_token = torch.compile(
                    utils._generate_next_token_no_sample,
                    mode="max-autotune",
                    fullgraph=True
                )

                t0 = time.perf_counter()
                _ = utils._generate(
                    model=self._model,
                    prompt=prompt_tokens,
                    max_generated_tokens=2,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    stop_tokens=self._tokenizer.stop_tokens,
                    custom_generate_next_token=custom_generate_next_token,
                )
                t = time.perf_counter() - t0                
                _has_compiled = True
                if self._is_rank_zero:
                    logger.info(f"Warmup run for compiled model takes: {t:.02f} sec")
            ###########################  COMPILATION  ###############################
            
            # NOTE: small hack to avoid the progress bar when running with `torch.compile`
            pbar = pbar or tqdm(total=len(self._train_dataloader), disable=not (rank == 0))
            t0   = time.perf_counter()
            
            generated_tokens = utils._generate(
                model=self._model,
                prompt=prompt_tokens,
                max_generated_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            pbar.update(1)

            # TODO: cache this number even if we don't use it
            model_size = sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(
                        self._model.parameters(), self._model.buffers()
                    )
                ]
            )            
            
            tokens_generated = len(generated_tokens[0]) - prompt_tokens.size(0)
            tokens_sec       = tokens_generated / t
            prompt_string    = self._tokenizer.decode(prompt_tokens.tolist())
            generated_string = self._tokenizer.decode(generated_tokens[0][-tokens_generated:])
            expected_string  = self._tokenizer.decode(labels[~is_prompt_mask].tolist())
            
            if self._cfg.get("log_outputs", False):
                generation_log_string = (
                    f"\n"
                    f"Generated tokens: {self._tokenizer.decode(generated_tokens[0][-tokens_generated:])}\n"
                    f"Expected tokens:  {self._tokenizer.decode(labels[~is_prompt_mask].tolist())}"
                    f"\n"
                )

                logger.info(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
                logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
                logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
                logger.info(generation_log_string)

            # store the generated tokens in the worker buffer
            rank_results.append({
                "prompt":    prompt_string,
                "generated": generated_string,
                "expected":  expected_string,
            })
        
        # NOTE: fix this for multi-node settings
        #       we need to gather the results from all the ranks
        with open(f"rank_{rank}_results.json", "w") as f:
            json.dump(rank_results, f)

        logger.info(f"Rank {rank} finished generating tokens.")
        logger.info(f"Rank {rank} results are saved at rank_{rank}_results.json")
        torch.distributed.barrier()

        if self._is_rank_zero:
            # merge all the results from different ranks
            all_results = []
            for i in range(world_size):
                with open(f"rank_{i}_results.json", "r") as f:
                    all_results.extend(json.load(f))
                
                # remove the file after merging
                os.remove(f"rank_{i}_results.json")

            # save the merged results with the checkpoint name as the filename
            checkpoint_name = cfg.checkpointer.checkpoint_files[0]
            # the dir is the last part of the checkpoint dir from the config
            checkpoint_dir  = os.path.basename(os.path.normpath(cfg.checkpointer.checkpoint_dir))
            checkpoint_name = f"{checkpoint_dir}_{checkpoint_name}"
            with open(f"{checkpoint_name}_results.jsonl", "w") as f:
                for result in all_results:
                    json.dump(result, f)
                    f.write('\n')


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
