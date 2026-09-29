import os

from huggingface_hub import hf_hub_download

# Anchor to the repository root (parent of tools/) so that the
# download lands in <repo_root>/pretrained_models regardless of CWD
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_id = 'XXH333/HD-Emo'
local_dir = os.path.join(repo_root, 'pretrained_models', 'HD-Emo')

# hd-emo codec checkpoint (a single .pt file)
model_path = hf_hub_download(
    repo_id=repo_id,
    filename='model.pt',
    local_dir=local_dir
)

print(f"HD-Emo codec checkpoint downloaded to '{model_path}'")
