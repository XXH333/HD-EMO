import os

from modelscope import snapshot_download

# Anchor to the repository root (parent of tools/) so that the
# download lands in <repo_root>/pretrained_models regardless of CWD
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_dir = os.path.join(repo_root, 'pretrained_models', 'CosyVoice2-0.5B')

snapshot_download('iic/CosyVoice2-0.5B', local_dir=local_dir)

print(f"CosyVoice2-0.5B downloaded to '{local_dir}'")
