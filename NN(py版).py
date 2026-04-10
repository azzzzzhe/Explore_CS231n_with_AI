import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ========== 0. 全局配置与设备检查 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

batch_size = 128
num_epochs = 10

resnet_mean = [0.485, 0.456, 0.406]
resnet_std  = [0.229, 0.224, 0.225]

# ========== 1. 数据预处理与加载 ==========


train_dataset = datasets.CIFAR10(root='C:/Jupyter(Anaconda)/data', train=True, download=False)
test_dataset = datasets.CIFAR10(root='C:/Jupyter(Anaconda)/data', train=False, download=False)



# ========== 2. 特征提取器 ==========
resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 移除最后的全连接层
resnet_model.fc = nn.Identity()
resnet_model = resnet_model.to(device)
resnet_model.eval()

# 冻结特征提取器的参数
for param in resnet_model.parameters():
    param.requires_grad = False

#特征提取函数

@torch.no_grad()
def fast_extract(imgs_np, batch_size=1024):
    """
    最快方案：
    - 一次性把全部图像转为 Tensor
    - 直接在 GPU 上分批推理，无 DataLoader 开销
    - FP16 autocast 加速
    - 预分配输出 tensor
    """
    resnet_model.eval()

    all_imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float().div_(255.0)

    # 提前移到 GPU，循环内不再重复 .to(device)
    mean = torch.tensor(resnet_mean).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(resnet_std).view(1, 3, 1, 1).to(device)

    N = all_imgs.shape[0]
    all_features = torch.zeros(N, 512, dtype=torch.float32)

    for i in range(0, N, batch_size):
        batch = all_imgs[i:i + batch_size].to(device)
        batch = F.interpolate(batch, size=(224, 224),
                              mode='bilinear', align_corners=False)
        batch.sub_(mean).div_(std)          # in-place 归一化，避免新建临时 tensor

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            feat = resnet_model(batch)

        all_features[i:i + batch_size] = feat.float().cpu()

    return all_features


print("正在提取数据集特征...")
X_train = fast_extract(train_dataset.data)
X_test = fast_extract(test_dataset.data)

y_train=torch.tensor(train_dataset.targets)
y_test=torch.tensor(test_dataset.targets)

#标准化
feat_mean = X_train.mean(dim=0)
feat_std  = X_train.std(dim=0) + 1e-8          #防止除零

X_train = (X_train - feat_mean) / feat_std
X_test  = (X_test  - feat_mean) / feat_std     #必须用训练集统计量归一化测试集

# ─────────────────────────────────────────────
# DataLoader（drop_last=True 防止 BN 在 batch=1 时崩溃）
# ─────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True,
    drop_last=True,                 # 丢弃最后不足 batch_size 的样本
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=256,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)
# ========== 3. 分类器（无 Dropout 版 MLP）==========
# 结构更加紧凑，仅包含线性层和激活函数
classifier = nn.Sequential(
    nn.Linear(512, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 10)  
).to(device)

#评估函数
def evaluate(test_loader):
    """返回在给定 loader 上的准确率"""
    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = classifier(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)
    return correct / total

# ========== 4. 优化器与损失函数 ==========
# 【优化核心】：在这里加入了 weight_decay=1e-4（L2正则化）来替代 Dropout 防止过拟合
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ========== 5. 训练与评估循环 ==========
print("开始训练...")
for epoch in range(num_epochs):
    # --- 训练阶段 ---
    classifier.train() 
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = classifier(images)
        loss = loss_fn(outputs, labels)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计训练指标
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * train_correct / train_total
    
    
    
    # 打印日志
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
    )

print("训练完成！")

# ========== 6. 最终评估 ==========
print("正在评估测试集...")
test_acc = evaluate(test_loader)
print(f"测试集准确率: {test_acc*100:.2f}%")#90%左右