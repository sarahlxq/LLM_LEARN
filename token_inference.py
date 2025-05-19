from tokenizers import (decoders, models, normalizers, pre_tokenizers,\
    processors, Tokenizer
)

tokenizer = Tokenizer.from_file("token_dir/tokenizer.json")
text = "I have a dog123#"
print(tokenizer.encode(text).tokens)

ids = tokenizer.encode(text).ids
print(ids)
print(tokenizer.decode(ids))
