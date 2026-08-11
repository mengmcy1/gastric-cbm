# InfiniSplat P01 官方 RGB 冒烟可同步记录

状态：**待产出机器补齐（2026-08-11）**。

本目录只保存小型、可审计、可提交 Git 的复现证据，不保存 checkpoint、PLY、MP4、逐帧图像或日志。

## 已知正式运行

- `outputs/P01_official_rgb_ply_smoke_v1/`
- `outputs/P01_official_rgb_video_smoke_v1/`

它们目前只存在于完成 Windows/RTX 5060 Laptop 复现的另一台电脑。当前电脑不得根据进度文档中的省略哈希补写正式清单。

## 产出机器需要补齐

- `artifact_manifest.json`：两次运行的运行 ID、产出机器简称、项目 Git commit、InfiniSplat 上游 commit、配置和输入相对路径、checkpoint/PLY/MP4 的字节数与完整 SHA256；
- `run_summary.json`：状态、起止时间或可核实耗时、峰值显存口径、高斯数量、警告和非确定性说明；
- `environment_lock.json`：OS、GPU、驱动、Python、PyTorch、CUDA、torchvision、xformers、gsplat 版本，以及补丁和编译后端哈希；
- `video_integrity.json`：编码、像素格式、宽高、FPS、时长、帧数、完整解码结果和完整 SHA256；
- `manual_review.json`：用户完整播放后再创建；未播放时不要用助手抽帧观察代填。

清单中的 SHA256 必须从原文件重新计算并完整写入。补齐后把本状态改为“已同步”，写明日期和对应 Git commit；大型产物仍由 `.gitignore` 排除。
