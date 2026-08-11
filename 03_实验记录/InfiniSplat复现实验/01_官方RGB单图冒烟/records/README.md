# InfiniSplat P01 官方 RGB 冒烟可同步记录

状态：**`win5060` 自动记录已补齐（2026-08-11）；用户完整播放仍待完成。**

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

`manual_review.json` 尚未创建，因为用户还没有完整播放官方 60 帧视频；不得用助手抽帧观察代填。大型产物仍由 `.gitignore` 排除，`linux5080` 只在开始 true_arc 适配时按 manifest 传输冻结候选 PLY。
