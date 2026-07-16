# SAE

SAE（Sparse Autoencoder，稀疏自编码器）用于从冻结分类模型的特征中发现可解释的
稀疏 feature。当前以 M-CBM 为主框架：先用 SAE 发现候选 feature，再由医生命名、
定义和标注概念，最后训练 Concept Bottleneck Layer 和稀疏癌/非癌分类器。

- `正式代码`：SAE 概念发现脚本及概念标注完成后的备用 M-CBM CBL 脚本。
- `前期训练`：字典规模、稀疏系数和目标层等参数实验。
- `demo测试`：复用正式 SAE 实现的少量患者功能验证入口。
