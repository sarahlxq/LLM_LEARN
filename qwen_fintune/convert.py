import json

# 系统提示语，可以根据需要修改
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
import json

def convert_json_to_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        # 读取整个 JSON 文件内容
        data = json.load(fin)

        # 遍历每个字典对象并写入 JSONL 文件
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"转换完成，结果已写入：{output_path}")


def convert_to_chat_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            data = json.loads(line.strip())

            instruction = data.get("instruction", "").strip()
            user_input = data.get("input", "").strip()
            output = data.get("output", "").strip()

            # 构造 messages 列表
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            if user_input:
                messages.append({"role": "user", "content": f"{instruction}\n\n{user_input}"})
            else:
                messages.append({"role": "user", "content": instruction})

            messages.append({"role": "assistant", "content": output})

            chat_data = {
                "messages": messages,
                "format": "chatml"
            }

            # 写入文件
            fout.write(json.dumps(chat_data, ensure_ascii=False) + "\n")

    print(f"转换完成，结果已写入：{output_path}")

# 示例调用
if __name__ == "__main__":
    input_json = "train_0001_of_0001.json"  
    output_jsonl = "train_0001_of_0001.jsonl"   # 输出文件路径
    convert_json_to_jsonl(input_json, output_jsonl)
    input_jsonl = "train_0001_of_0001.jsonl"     # 输入文件路径
    output_jsonl = "qwen2-sft-output.jsonl"
    convert_to_chat_format(input_jsonl, output_jsonl)
