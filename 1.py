import torch
import torch.nn as nn

# 创建一个示例模型
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init()
        self.embedding_layer1 = nn.Embedding(10000, 128)  # 嵌入层1
        self.embedding_layer2 = nn.Embedding(5000, 64)    # 嵌入层2
        self.fc = nn.Linear(256, 10)  # 全连接层

# 创建模型实例
model = ExampleModel()

# 计算嵌入参数总数
total_embedding_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total embedding parameters:", total_embedding_params)
