
import torch

# 假设文件名为 'model.pt'
# loaded_object = torch.load('/data/lh/python_code/edit_recommend/gradient/before.pt')
loaded_object = torch.load('/data/lh/python_code/edit_recommend/gradient/after.pt')

# 如果保存的是模型的状态字典
if isinstance(loaded_object, dict):
    # 这是一个状态字典
    print("Loaded a state dictionary.")
    
    # 查看所有层的名字和它们的参数
    for param_tensor in loaded_object:
        print(param_tensor, "\t", loaded_object[param_tensor].size())
        
    # 或者，如果你想要打印具体的参数值
    for key, value in loaded_object.items():
        print(f"Layer: {key}")
        print(value)  # 打印参数张量
        print("\n")

# 如果保存的是整个模型
elif isinstance(loaded_object, torch.nn.Module):
    # 这是一个完整的模型对象
    print("Loaded a complete model.")
    
    # 查看模型的结构
    print(loaded_object)
    
    # 查看模型的参数
    for name, param in loaded_object.named_parameters():
        print(f"Parameter: {name}, Size: {param.size()}")
        print(param)  # 打印参数张量
        print("\n")