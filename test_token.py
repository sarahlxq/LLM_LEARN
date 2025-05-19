#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tokenizers import (decoders, models, normalizers, pre_tokenizers,\
    processors, trainers, Tokenizer
)
from tqdm import tqdm
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
    pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

#text = "I'm eating apples.123"
#print(tokenizer.pre_tokenizer.pre_tokenize_str(text))
trainer = trainers.BpeTrainer(
    vocab_size=65535,
    special_tokens=["<|system|>","<|user|>", "<|assistant|>", "<|end|>"],
    min_frequency=1500,
)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

from datasets import load_dataset, Dataset
dataset: Dataset = load_dataset("json",data_files=[
    "/home/jovyan/lxq/finetune/dataset/part-000442-a894b46e.jsonl.tar.gz",
    "/home/jovyan/lxq/finetune/dataset/part-000419-a894b46e.jsonl.tar.gz"
    ],
    split="train",
    ignore_verifications=True
    )

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i:i+1000]["content"]


training_corpus = get_training_corpus()
progress_bar = tqdm(total=len(dataset) // 1000) # 假设每批次大小为1000

# 开始训练并更新进度条
tokenizer.train_from_iterator((texts for texts in training_corpus), trainer=trainer, length=len(dataset))
for _ in training_corpus:
    progress_bar.update(1)

progress_bar.close()

# 保存分词器
tokenizer.save("token_dir/tokenizer.json")

#tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)



