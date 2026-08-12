# InfiniSplat P01 官方 RGB 冒烟可同步记录

状态：**`win5060` 自动记录已补齐（2026-08-11）；`linux5080` 独立环境、PLY和官方视频冒烟记录已补齐（2026-08-12）；两台机器的用户完整播放均待完成。**

本目录只保存小型、可审计、可提交 Git 的复现证据，不保存 checkpoint、PLY、MP4、逐帧图像或日志。

## 已知正式运行

- `outputs/P01_official_rgb_ply_smoke_v1/`
- `outputs/P01_official_rgb_video_smoke_v1/`

它们由 `win5060` 生成，当前大型产物也位于 `win5060`。`linux5080` 是否持有副本以服务器本地核验为准。

## 已同步的小型证据

- `artifact_manifest.json`：两次运行、配置、输入、checkpoint、PLY 和 MP4 的完整字节数与 SHA256；
- `run_summary.json`：运行状态、近似耗时、显存抽样高点、高斯数量、警告和非确定性边界；
- `environment_lock.json`：`win5060` 的系统、GPU、驱动和独立 `infinisplat` 环境；
- `video_integrity.json`：编码、分辨率、帧率、帧数、时长、完整解码和完整 SHA256。

`linux5080` 的独立同配置复现保存在以下文件，不覆盖 Windows 记录：

- `linux5080_artifact_manifest.json`：服务器输入、权重、PLY和MP4的完整身份与冻结候选状态；
- `linux5080_environment_lock.json`：服务器Python/CUDA/PyTorch/gsplat与JIT扩展记录；
- `linux5080_run_summary.json`：PLY成功、两次视频诊断失败和最终视频成功的完整边界；
- `linux5080_video_integrity.json`：服务器官方60帧视频完整性。

`manual_review.json` 尚未创建，因为用户还没有完整播放官方 60 帧视频；不得用助手抽帧观察代填。大型产物仍由 `.gitignore` 排除。当前 Windows 和Linux的PLY都只是候选，必须由用户指定机器ID和完整SHA256后才能标记为冻结。

## Linux 服务器复现规则

`linux5080` 默认不迁移 Windows checkpoint、PLY、MP4 或环境。通过 Git 获取源码、`configs/`、`inputs/`、`scripts/` 和本 records；在独立 Linux `infinisplat` 环境重新下载并校验 checkpoint，然后运行：

```bash
python "03_实验记录/InfiniSplat复现实验/01_官方RGB单图冒烟/scripts/run_official_rgb_smoke.py" \
  --config "03_实验记录/InfiniSplat复现实验/01_官方RGB单图冒烟/configs/p01_official_rgb_video_smoke_v1.json" \
  --machine-id linux5080 \
  --run-suffix linux5080_YYYYMMDD
```

运行仍按 `configs/`、`inputs/`、`scripts/`、`records/`、`outputs/<运行ID>/` 和 `logs/` 分层；脚本默认拒绝覆盖，并在本地输出目录生成 `run_receipt.json`。服务器重跑的 PLY 是新的复现产物，必须用自己的 SHA256 登记；不能冒充 `win5060` 的 PLY。
