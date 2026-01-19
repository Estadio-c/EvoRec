import vllm
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import sys
import argparse
from vllm.lora.request import LoRARequest
# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = 608
# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.95 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


def generate(model_name,data_path,batch,max_new_tokens,save_name,lora_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    print('data的数量是{}'.format(len(data)))
    all_len = len(data)
    num = int(all_len/batch)
    llm = vllm.LLM(
        model_name,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        enable_lora=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for i in tqdm(range(num)):
        messages = []
        for j in range(i*batch, (i+1)*batch):

            message = [data[j]['messages'][0], data[j]['messages'][1]]
            messages.append(message)
        
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = vllm.SamplingParams(top_k=-1, top_p=0.9, temperature=0, max_tokens=max_new_tokens,seed=AICROWD_RUN_SEED)
        responses = llm.generate(prompts=inputs,use_tqdm = False, sampling_params=sampling_params,lora_request=LoRARequest("adapter", 1, lora_path))
        res = []
        for response in responses:
            res.append(response.outputs[0].text)
        with open(save_name, 'a') as f:
            for item in res:
                f.write(json.dumps(item) + '\n')
    if num*batch != all_len:
        messages = []
        for i in range(num*batch, all_len):

            message = [data[i]['messages'][0], data[i]['messages'][1]]
            messages.append(message)
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = vllm.SamplingParams(top_k=-1, top_p=0.8, temperature=0, max_tokens=max_new_tokens,seed=AICROWD_RUN_SEED)
        responses = llm.generate(prompts=inputs,use_tqdm = False, sampling_params=sampling_params,lora_request=LoRARequest("adapter2", 2, lora_path))
        res = []
        for response in responses:
            res.append(response.outputs[0].text)
        with open(save_name, 'a') as f:
            for item in res:
                f.write(json.dumps(item) + '\n')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--lora_path', type=str)
    args = parser.parse_args()
    generate(args.model_name,args.test_file,args.batch,args.max_len,args.save_name,args.lora_path)
