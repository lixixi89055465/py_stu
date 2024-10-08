import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内hf镜像
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from transformers import AutoTokenizer

# 加载BERT的分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

# 使用分词器处理文本
text = "你好，我在北京的互联网公司工作"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
"""
['你', '好', '，', '我', '在', '北', '京', '的', '互', '联', '网', '公', '司', '工', '作']
"""

# 或者直接编码为模型输入的格式
encoded_text = tokenizer.encode(text)
print(encoded_text)
"""
[101, 872, 1962, 8024, 2769, 1762, 1266, 776, 4638, 757, 5468, 5381, 1062, 1385, 2339, 868, 102]
"""

# 对于更复杂的输入准备，可以使用encode_plus
encoded_plus = tokenizer.encode_plus(text, add_special_tokens=True)
print(encoded_plus)
"""
{
    'input_ids': [101, 872, 1962, 8024, 2769, 1762, 1266, 776, 4638, 757, 5468, 5381, 1062, 1385, 2339, 868, 102], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
"""
print("test")