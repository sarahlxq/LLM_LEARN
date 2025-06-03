import torch
import numpy as np
import random

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



#set_random_seed(4)
#print(random.random())            # 输出固定值
#print(np.random.rand(2))          # 输出固定数组
#print(torch.rand(2))



logits = torch.tensor([100.0, -50.0, 0.0])
soft_cap = 30.0

# Soft cap with tanh
logits_capped = torch.tanh(logits / soft_cap) * soft_cap
print(logits_capped)