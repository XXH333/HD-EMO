import os

from huggingface_hub import snapshot_download

# Anchor to the repository root (parent of tools/) so that the
# download lands in <repo_root>/pretrained_models regardless of CWD
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_id = 'openai/whisper-small'
local_dir = os.path.join(repo_root, 'pretrained_models', 'whisper')

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    max_workers=3 # limit download concurrency
)

print(f"'{repo_id}' downloaded to '{local_dir}'")
