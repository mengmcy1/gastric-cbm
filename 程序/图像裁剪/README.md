# 图像预处理

从原始胃镜图到学生输入的操作顺序：

1. brightness_crop_v2.py：亮度轮廓及四边裁剪。
2. 根据mapping.csv中的复核状态检查误裁；需要时使用dark_crop_safety_review.py。
3. fov_mask_v1.py：使用shiye_V1.onnx提取有效视野并遮黑视野外区域，保留当前尺寸。
4. 在最终图片坐标系准备病灶框和教师ROI；模型内部再resize到224×224。

完整命令和环境说明见[项目说明](../../README.md)。

已有交接的正式输入图无需再次裁边。Keep流程保留原约定；不要自动改为去画中画notch分支。pip_detector.py等是候选检测和人工复核工具，不自动决定是否删除画中画区域。

默认FOV阈值0.5、内缩1像素；新来源应检查预览与pending记录，不能根据测试集表现调整预处理参数。ONNX权重从私有Release获取，Python后端使用onnxruntime CPU，也保留OpenCV DNN后端。

参考模板只提供列结构，不含患者资料。
