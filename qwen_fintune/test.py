from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/home/jovyan/lxq/model_dir/Qwen/Qwen2___5-0___5B-Instruct")
for name, module in model.named_modules():
    print(name)