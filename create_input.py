num_seqs = 1  # 序列的数量 batch_size
num_heads = 2
num_kv_heads = 2
head_size = 64
block_size = 16 # 表示每个block存放的token总数
num_blocks = 2
max_seq_len = 32
device = torch.device("cuda")

def create_test_inputs():
    query = torch.arange(0, num_seqs * num_heads * head_size, dtype=torch.float32, device=device).reshape(num_seqs, num_heads, head_size)
    value_cache = torch.arange(0, num_blocks * num_kv_heads * head_size * block_size,  dtype=torch.float32, device=device).reshape(num_blocks, block_size, num_kv_heads, head_size).permute(0,2,3,1).contiguous()
    key_cache = torch.arange(0, num_blocks * num_kv_heads * head_size * block_size,  dtype=torch.float32, device=device).reshape(num_blocks, block_size, num_kv_heads, head_size // 4, 4).permute(0,2,3,1,4).contiguous()

    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.zeros((num_seqs, max_blocks_per_seq), dtype=torch.int32, device=device)  # 只用第一个block
    seq_lens = torch.tensor([block_size, block_size], dtype=torch.int32, device=device)  # 使用全部 16 token

    return query, key_cache, value_cache, block_tables, seq_lens