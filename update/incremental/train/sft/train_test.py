# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from accelerate import Accelerator
from torch.utils.data import Dataset,DataLoader
from typing import TYPE_CHECKING, List, Optional
import torch
from prompt import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from prompt import IGNORE_INDEX
from prompt import get_logits_processor
# from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
# from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
from transformers import Trainer
from peft import PeftModel,PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM
)
import json
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def train_try(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):     
    accelerator = Accelerator() 
    device = accelerator.device
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    # print(dataset_module['train_dataset'][0])
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # model.load_adapter(finetuning_args.adapter_path, 'default', is_trainable=True)
    
    print('成功加载',finetuning_args.adapter_path)
    # model.set_adapter(["default"])

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    #['train_dataset', 'eval_dataset']


    train_loader = DataLoader(dataset_module['train_dataset'], batch_size=1, shuffle=True,num_workers=4, collate_fn=data_collator)
    optimizer = torch.optim.AdamW([param for name, param in model.named_parameters() if 'lora' in name], lr=3e-4)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    

    #记录总步数
    step = 0

    for epoch in range( 2):
        model.train()
        
        accelerator.wait_for_everyone()

        # retain 训练
        for forget_batch in train_loader:
            step += 1
            model.train()
            optimizer.zero_grad()
            print(forget_batch['input_ids'].shape)
            outputs = model(**forget_batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            print(1111)