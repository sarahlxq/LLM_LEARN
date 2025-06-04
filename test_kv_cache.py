import argparse

def estimate_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "float16",
    gqa: bool = False,
    num_key_value_heads: int = None,
    verbose: bool = True
):
    """
    估算 KV Cache 的显存占用。
    
    参数说明：
    - batch_size: 并发请求数
    - max_seq_len: 最大序列长度（prompt + 生成 token）
    - num_layers: 模型层数（如 LLaMA-7B 是 32 层）
    - num_heads: 每层的注意力头数（对于 GQA，这是 query heads 数量）
    - head_dim: 每个 attention head 的维度（如 128）
    - dtype: 数据类型，支持 float16 / bfloat16 / float32
    - gqa: 是否使用 Grouped Query Attention
    - num_key_value_heads: GQA 中 Key/Value 的头数量（如 8）
    - verbose: 是否打印详细信息
    """

    # 数据类型字节数
    dtype_size_map = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }
    if dtype not in dtype_size_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    dtype_size = dtype_size_map[dtype]

    # 如果使用 GQA，则 key/value 的 head 数小于 query 的 head 数
    if gqa:
        if num_key_value_heads is None:
            raise ValueError("GQA 需要提供 num_key_value_heads")
        kv_heads = num_key_value_heads
    else:
        kv_heads = num_heads

    # 单个 token 的 KV 缓存大小（每个 layer）
    per_token_kv_size_per_layer = 2 * kv_heads * head_dim  # K + V

    # 每个请求每层的最大缓存大小
    per_request_per_layer_kv = max_seq_len * per_token_kv_size_per_layer

    # 总体 KV Cache 大小（所有 batch + 所有层）
    total_kv_cache_bytes = (
        batch_size *
        num_layers *
        per_request_per_layer_kv *
        dtype_size
    )

    # 转换为 GB
    total_kv_cache_gb = total_kv_cache_bytes / (1024 ** 3)

    if verbose:
        print("\n📊 KV Cache 显存估算结果:")
        print(f"  - 并发请求数 (batch_size): {batch_size}")
        print(f"  - 最大序列长度 (max_seq_len): {max_seq_len}")
        print(f"  - 模型层数 (num_layers): {num_layers}")
        print(f"  - 注意力头数 (num_heads): {num_heads}")
        if gqa:
            print(f"  - Key/Value 头数 (num_key_value_heads): {num_key_value_heads}")
        print(f"  - 每头维度 (head_dim): {head_dim}")
        print(f"  - 数据类型 (dtype): {dtype}")
        print(f"\n🧠 总计 KV Cache 显存占用 ≈ {total_kv_cache_gb:.2f} GB")

    return total_kv_cache_gb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="估算 KV Cache 显存占用")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gqa", action="store_true", help="是否启用 GQA")
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    args = parser.parse_args()

    estimate_kv_cache(**vars(args))
