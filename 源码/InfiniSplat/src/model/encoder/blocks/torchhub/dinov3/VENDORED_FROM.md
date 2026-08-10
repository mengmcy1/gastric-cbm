# Vendored DINOv3

- Upstream repository: https://github.com/facebookresearch/dinov3.git
- Pinned commit: `6e50ab28b75133230b6cc3a846a8adc3c300b27d`
- Import date: `2026-03-27`
- Scope: `InfiniSplat image branch only`
- Notes: Imported from the official upstream repository into this workspace snapshot. The nested `.git` directory is intentionally removed.
- Local patch: `dinov3/hub/backbones.py` accepts local `dinov3_vitl16` checkpoint paths without the upstream `-<8char_hash>.pth` suffix and infers the correct `untie_global_and_local_cls_norm` mode from the filename.
- Local patch: `dinov3/hub/backbones.py` loads local checkpoint paths with `torch.load(...)` directly instead of converting them to `file://` URLs and copying them into `~/.cache/torch/hub`.
