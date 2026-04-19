# -*- coding: utf-8 -*-
"""
模型训练脚本
实现类别加权交叉熵、学习率衰减、早停等策略
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from model import ActionRecognitionModel, PureLSTMModel, count_parameters
from config import *


def load_dataset(split_name):
    """加载指定分割的数据集"""
    split_dir = os.path.join(DATA_DIR, split_name)
    samples = np.load(os.path.join(split_dir, "samples.npy"))
    labels = np.load(os.path.join(split_dir, "labels.npy"))
    return samples, labels


def compute_class_weights(labels):
    """计算类别权重（与样本数成反比）"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    weight_tensor = torch.zeros(NUM_CLASSES)
    for cls_idx, w in zip(unique, weights):
        weight_tensor[cls_idx] = w
    return weight_tensor


def train_model(model, train_loader, val_loader, class_weights, device,
                model_name="cnn_lstm", save_dir=MODEL_DIR):
    """训练模型"""
    os.makedirs(save_dir, exist_ok=True)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    # 训练记录
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f"best_{model_name}.pth")

    print(f"\n开始训练 {model_name}...")
    print(f"参数量: {count_parameters(model):,}")
    print(f"设备: {device}")
    print(f"最大轮数: {MAX_EPOCHS}")
    print("-" * 60)

    for epoch in range(MAX_EPOCHS):
        # ============ 训练阶段 ============
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ============ 验证阶段 ============
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss_sum += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        # 记录
        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # 学习率调度
        scheduler.step(val_loss)

        # 打印进度
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  -> 保存最佳模型 (验证准确率: {val_acc:.4f})")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发！最佳模型在 Epoch {best_epoch}，验证准确率: {best_val_acc:.4f}")
            break

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # 保存训练历史
    history_path = os.path.join(save_dir, f"history_{model_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history, best_model_path


def plot_training_curves(history, model_name="cnn_lstm", save_dir=MODEL_DIR):
    """绘制训练过程曲线（用于论文图5.1）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 损失曲线
    axes[0].plot(epochs, history["train_loss"], 'b-', linewidth=1.5, label='训练损失')
    axes[0].plot(epochs, history["val_loss"], 'r-', linewidth=1.5, label='验证损失')
    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
    axes[0].set_ylabel('损失值 (Loss)', fontsize=12)
    axes[0].set_title('训练与验证损失变化曲线', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, history["train_acc"], 'b-', linewidth=1.5, label='训练准确率')
    axes[1].plot(epochs, history["val_acc"], 'r-', linewidth=1.5, label='验证准确率')
    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
    axes[1].set_ylabel('准确率 (Accuracy)', fontsize=12)
    axes[1].set_title('训练与验证准确率变化曲线', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_curves_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def main():
    """主训练入口"""
    print("=" * 60)
    print("  模型训练")
    print("=" * 60)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据集...")
    train_samples, train_labels = load_dataset("train")
    val_samples, val_labels = load_dataset("verify")

    print(f"训练集: {train_samples.shape} 标签: {train_labels.shape}")
    print(f"验证集: {val_samples.shape} 标签: {val_labels.shape}")

    # 获取特征维度
    feature_dim = train_samples.shape[2]
    print(f"特征维度: {feature_dim}")

    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(train_samples),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_samples),
        torch.LongTensor(val_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 类别权重
    class_weights = compute_class_weights(train_labels)
    print(f"类别权重: {class_weights}")

    # ============ 训练 1D-CNN + LSTM 模型 ============
    print("\n" + "=" * 40)
    print("  训练 1D-CNN + LSTM 模型")
    print("=" * 40)
    cnn_lstm_model = ActionRecognitionModel(input_dim=feature_dim).to(device)
    history_cnn, best_path_cnn = train_model(
        cnn_lstm_model, train_loader, val_loader, class_weights, device,
        model_name="cnn_lstm"
    )
    plot_training_curves(history_cnn, model_name="cnn_lstm")

    # ============ 训练纯 LSTM 模型（对比实验）============
    print("\n" + "=" * 40)
    print("  训练纯 LSTM 模型 (对比实验)")
    print("=" * 40)
    lstm_model = PureLSTMModel(input_dim=feature_dim).to(device)
    history_lstm, best_path_lstm = train_model(
        lstm_model, train_loader, val_loader, class_weights, device,
        model_name="pure_lstm"
    )
    plot_training_curves(history_lstm, model_name="pure_lstm")

    print("\n" + "=" * 60)
    print("  训练完成！")
    print("=" * 60)
    print(f"1D-CNN + LSTM 最佳模型: {best_path_cnn}")
    print(f"纯 LSTM 最佳模型: {best_path_lstm}")


if __name__ == "__main__":
    main()
