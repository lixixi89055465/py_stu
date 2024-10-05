from huggingface_hub import hf_hub_download

# 下载模型文件并保存到本地
file_path = hf_hub_download(repo_id="bigscience/bloom-560m", filename="pytorch_model.bin", local_dir="bloom-560m")

print(f"Model file downloaded to: {file_path}")
