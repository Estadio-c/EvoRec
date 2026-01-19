# An Efficient LLM-based Evolutional Recommendation with Locate-Forget-Update Paradigm

This repository provides a framework for training, fine-tuning, and evaluating EvoRec, an incremental large language model (LLM) recommendation system. The following steps outline the complete workflow.

data: https://pan.baidu.com/s/1OlnlAErfXWZZMJ3gVSHKag Extract code: h6du 
## Workflow

1. **Train the Baseline LLMRec Model**  
Train the original LLM-based recommendation model using:  
   ```sh
   cd EvoRec  
   sh train_sft.sh  
2. **The original LLM performs inference on the data updated with user interactions.**  
   ```sh
   cd inference_all  
   sh vllm_lora_sft.sh
3. **Identify Sensitive Parameters**  
Locate the model parameters that are most sensitive to outdated or biased knowledge:
   ```sh
   sh localization.sh
1. **Forget Outdated Information**  
Perform an unlearning process to remove outdated data while preserving useful knowledge::
   ```sh
   sh unlearning.sh
1. **Identify the Best Checkpoint on the Validation Set**  
Find the optimal checkpoint based on validation performance:
   ```sh
   cd inference_update && sh vllm_lora_edit.sh
1. **Generate the final recommendation results using the updated recommendation system.**  
Run inference on the complete dataset to obtain the final recommendation results:
   ```sh
   cd ../inference_all && sh vllm_lora_edit.sh