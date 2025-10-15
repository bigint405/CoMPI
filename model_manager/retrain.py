import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

from datasets.imagenet_utils import CustomImageFolder

def retrain_with_task(model, target_classes, train_data_dirs, val_data_dirs, device_num, log_dir, task_name="", num_epochs=20, batch_size=128, lr=0.01):
    # 设置随机种子
    seed = 42
    torch.cuda.set_device(device_num)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载训练和验证数据集
    print(f"{task_name} start load train_dataset")
    train_dataset = CustomImageFolder(train_data_dirs, target_classes, transform=transform)
    train_dataset.load()
    print(f"{task_name} finish load train_dataset")
    print(f"{task_name} start load val_dataset")
    val_dataset = CustomImageFolder(val_data_dirs, target_classes, transform=transform)
    val_dataset.load()
    print(f"{task_name} finish load val_dataset")

    # 定义 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 将模型移动到设备上
    device = f"cuda:{device_num}"
    model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    losses = []
    accuracies = []

    os.makedirs(log_dir, exist_ok=True)

    # 训练模型
    for epoch in range(0, num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 在验证集上测试模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            accuracies.append(accuracy)
            print(f'{task_name} Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {100 * accuracy:.2f}%')

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}.pth'))
    del images, labels
    model.to("cpu")
    # 绘制损失和准确率曲线
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, num_epochs+1), losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, num_epochs+1), accuracies, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy')
    plt.savefig(os.path.join(log_dir, 'loss_and_acc.png'))

    # 保存日志
    with open(os.path.join(log_dir, 'loss_and_acc.log'), 'w') as f:
        f.write("losses:\n")
        formatted_losses = [f"{loss:.4g}" for loss in losses]  # 保留4位有效数字
        f.write("\n".join(formatted_losses))  # 每个损失值占一行
        f.write("\naccuracies:\n")
        formatted_accuracies = [f"{acc:.4g}" for acc in accuracies]  # 保留4位有效数字
        f.write("\n".join(formatted_accuracies))  # 每个准确率值占一行
        f.write("\n")
