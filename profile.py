import torch

free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
print("free_gpu_memory: ",free_gpu_memory, "total_gpu_memory: ", total_gpu_memory)