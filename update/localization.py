import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Locate sensitive layers in a model.")
    parser.add_argument("--data_before", type=str, required=True, help="Path to edit_before_candidate_train.json")
    parser.add_argument("--data_after", type=str, required=True, help="Path to edit_after_candidate_train.json")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to the LoRA checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save located sensitive layers")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=1200, help="Maximum input sequence length")
    return parser.parse_args()

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def locate_sensitive_layer(model, tokenizer, max_length, target_new_list, ground_truth_list, batch_size):
    assert len(target_new_list) == len(ground_truth_list)
    toxic_layer = []
    num_samples = len(target_new_list)
    
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_target_new = target_new_list[start_idx:end_idx]
        batch_ground_truth = ground_truth_list[start_idx:end_idx]

        batch_target_new = [tokenizer.apply_chat_template([item['messages'][0], item['messages'][1]], tokenize=False, add_generation_prompt=True) for item in batch_target_new]
        batch_ground_truth = [tokenizer.apply_chat_template([item['messages'][0], item['messages'][1]], tokenize=False, add_generation_prompt=True) for item in batch_ground_truth]

        input = tokenizer(
            [item for pair in zip(batch_target_new, batch_ground_truth) for item in pair],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)
       
        with torch.no_grad():
            outputs = model(**input)
        hidden_states = outputs.hidden_states
    
        for j in range(len(batch_target_new)):
            max_distance_layer = None
            max_distance_value = float('inf')
            for layer_index in range(1, len(hidden_states)):
                pool = torch.sum(hidden_states[layer_index], dim=1) / hidden_states[layer_index].size(1)
                cosine_sim_single = F.cosine_similarity(pool[j * 2].unsqueeze(0), pool[j * 2 + 1].unsqueeze(0), dim=1)
                cosine_sim_scalar = cosine_sim_single.item()
                if cosine_sim_scalar < max_distance_value:
                    max_distance_value = cosine_sim_scalar
                    max_distance_layer = layer_index
            toxic_layer.append(max_distance_layer - 1)
        
        del outputs
        del hidden_states
        torch.cuda.empty_cache()
    
    counter = Counter(toxic_layer)
    top_9 = counter.most_common(9)
    print(top_9)
    return [item for item, _ in top_9]

def main():
    args = parse_args()
    
    # 加载数据
    data_before = load_data(args.data_before)
    data_after = load_data(args.data_after)
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}, output_hidden_states=True
    )
    peft_model = PeftModel.from_pretrained(model, args.lora_checkpoint)
    
    # 定位敏感层
    sensitive_layer = locate_sensitive_layer(peft_model, tokenizer, args.max_length, data_before, data_after, args.batch_size)
    
    # 目标层
    target_patterns = [
        "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.default.weight",
        "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.default.weight",
        "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.default.weight",
        "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.default.weight",
        "base_model.model.model.layers.{}.mlp.gate_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.mlp.gate_proj.lora_B.default.weight",
        "base_model.model.model.layers.{}.mlp.up_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.mlp.up_proj.lora_B.default.weight",
        "base_model.model.model.layers.{}.mlp.down_proj.lora_A.default.weight",
        "base_model.model.model.layers.{}.mlp.down_proj.lora_B.default.weight"
    ]
    
    target_layer = [pattern.format(layer) for layer in sensitive_layer for pattern in target_patterns]
    
    # 保存结果
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(target_layer, f, ensure_ascii=False, indent=4)
    
    print(f"Located layers saved to {args.output_path}")

if __name__ == "__main__":
    main()
