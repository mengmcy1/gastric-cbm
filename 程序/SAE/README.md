# SAE

> 2026-08-19状态：此前ResNet50与EfficientNet-B0全局/局部SAE路线已冻结归档，旧代码与
> 正式结果为保证可追溯而原地保留。当前文献驱动的C-long SAE新路线见根目录
> `SAE实验进度与结果讨论.md`；新路线尚未形成正式代码入口，不能把下列旧命令直接当作
> C-long实验命令。旧路线索引见`程序/归档/SAE/旧路线索引_20260819.md`。

SAE（Sparse Autoencoder，稀疏自编码器）用于从冻结分类模型的特征中发现可解释的
稀疏 feature。当前以 M-CBM 为主框架：先用 SAE 发现候选 feature，再由医生命名、
定义和标注概念，最后训练 Concept Bottleneck Layer 和稀疏癌/非癌分类器。

- `正式代码`：SAE 概念发现脚本及概念标注完成后的备用 M-CBM CBL 脚本。
- `程序/归档/SAE/`：字典规模实验、前期说明和少量患者demo。
