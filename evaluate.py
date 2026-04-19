# -*- coding: utf-8 -*-
"""
模型评估脚本
生成混淆矩阵、分类报告
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, average_precision_score)
from model import ActionRecognitionModel, PureLSTMModel
from config import *


def load_test_data():
    """加载测试集"""
    test_dir = os.path.join(DATA_DIR, "test")
    samples = np.load(os.path.join(test_dir, "samples.npy"))
    labels = np.load(os.path.join(test_dir, "labels.npy"))
    return samples, labels


def evaluate_model(model, samples, labels, device):
    """评估模型，返回预测结果"""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        # 分批处理
        batch_size = 64
        for i in range(0, len(samples), batch_size):
            batch = torch.FloatTensor(samples[i:i + batch_size]).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """绘制混淆矩阵"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='真实标签',
           xlabel='预测标签')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子中显示数值
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_pr_curves(y_true, y_probs, class_names, title, save_path):
    """绘制多类别 Precision-Recall 曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    n_classes = len(class_names)
    # one-hot 编码真实标签
    y_true_onehot = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f'{class_names[i]} (AP={ap:.3f})')

    # 计算 micro-average
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_onehot.ravel(), y_probs.ravel())
    ap_micro = average_precision_score(y_true_onehot, y_probs, average='micro')
    ax.plot(recall_micro, precision_micro, color='black', lw=2, linestyle='--',
            label=f'micro-avg (AP={ap_micro:.3f})')

    ax.set_xlabel('Recall (召回率)', fontsize=12)
    ax.set_ylabel('Precision (精确率)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR 曲线已保存: {save_path}")


def main():
    """主评估入口"""
    print("=" * 60)
    print("  模型评估与对比实验")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_samples, test_labels = load_test_data()
    feature_dim = test_samples.shape[2]
    print(f"测试集: {test_samples.shape}, 标签分布: {np.bincount(test_labels)}")

    class_names_cn = [ACTION_NAMES_CN[c] for c in ACTION_CLASSES]
    results = {}

    # ============ 评估 1D-CNN + LSTM ============
    print("\n" + "=" * 40)
    print("  评估 1D-CNN + LSTM 模型")
    print("=" * 40)

    cnn_lstm_path = os.path.join(MODEL_DIR, "best_cnn_lstm.pth")
    if os.path.exists(cnn_lstm_path):
        cnn_lstm_model = ActionRecognitionModel(input_dim=feature_dim).to(device)
        checkpoint = torch.load(cnn_lstm_path, map_location=device, weights_only=False)
        cnn_lstm_model.load_state_dict(checkpoint['model_state_dict'])

        preds_cnn, probs_cnn = evaluate_model(cnn_lstm_model, test_samples, test_labels, device)

        # 分类报告
        acc_cnn = accuracy_score(test_labels, preds_cnn)
        f1_macro = f1_score(test_labels, preds_cnn, average='macro')
        report = classification_report(test_labels, preds_cnn,
                                       target_names=class_names_cn,
                                       output_dict=True)
        print(f"\n总体准确率: {acc_cnn:.4f}")
        print(f"宏平均 F1: {f1_macro:.4f}")
        print("\n详细分类报告:")
        print(classification_report(test_labels, preds_cnn, target_names=class_names_cn))

        # 混淆矩阵
        plot_confusion_matrix(
            test_labels, preds_cnn, class_names_cn,
            "1D-CNN + LSTM 模型混淆矩阵",
            os.path.join(MODEL_DIR, "confusion_matrix_cnn_lstm.png")
        )

        # PR 曲线
        plot_pr_curves(
            test_labels, probs_cnn, class_names_cn,
            "1D-CNN + LSTM 模型 Precision-Recall 曲线",
            os.path.join(MODEL_DIR, "pr_curve_cnn_lstm.png")
        )

        results["cnn_lstm"] = {
            "accuracy": float(acc_cnn),
            "f1_macro": float(f1_macro),
            "per_class": {}
        }
        for i, cls_name in enumerate(ACTION_CLASSES):
            cn_name = ACTION_NAMES_CN[cls_name]
            if cn_name in report:
                results["cnn_lstm"]["per_class"][cls_name] = {
                    "precision": report[cn_name]["precision"],
                    "recall": report[cn_name]["recall"],
                    "f1": report[cn_name]["f1-score"],
                    "support": report[cn_name]["support"]
                }
    else:
        print(f"[警告] 1D-CNN + LSTM 模型未找到: {cnn_lstm_path}")

    # ============ 评估纯 LSTM ============
    print("\n" + "=" * 40)
    print("  评估纯 LSTM 模型 (对比实验)")
    print("=" * 40)

    lstm_path = os.path.join(MODEL_DIR, "best_pure_lstm.pth")
    if os.path.exists(lstm_path):
        lstm_model = PureLSTMModel(input_dim=feature_dim).to(device)
        checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
        lstm_model.load_state_dict(checkpoint['model_state_dict'])

        preds_lstm, probs_lstm = evaluate_model(lstm_model, test_samples, test_labels, device)

        acc_lstm = accuracy_score(test_labels, preds_lstm)
        f1_lstm = f1_score(test_labels, preds_lstm, average='macro')
        print(f"\n总体准确率: {acc_lstm:.4f}")
        print(f"宏平均 F1: {f1_lstm:.4f}")
        print("\n详细分类报告:")
        print(classification_report(test_labels, preds_lstm, target_names=class_names_cn))

        # 混淆矩阵
        plot_confusion_matrix(
            test_labels, preds_lstm, class_names_cn,
            "纯 LSTM 模型混淆矩阵",
            os.path.join(MODEL_DIR, "confusion_matrix_pure_lstm.png")
        )

        # PR 曲线
        plot_pr_curves(
            test_labels, probs_lstm, class_names_cn,
            "纯 LSTM 模型 Precision-Recall 曲线",
            os.path.join(MODEL_DIR, "pr_curve_pure_lstm.png")
        )

        results["pure_lstm"] = {
            "accuracy": float(acc_lstm),
            "f1_macro": float(f1_lstm),
        }
    else:
        print(f"[警告] 纯 LSTM 模型未找到: {lstm_path}")

    # ============ 保存评估结果 ============
    results_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存: {results_path}")

    # ============ 对比总结 ============
    if "cnn_lstm" in results and "pure_lstm" in results:
        print("\n" + "=" * 40)
        print("  模型对比总结")
        print("=" * 40)
        print(f"1D-CNN + LSTM 准确率: {results['cnn_lstm']['accuracy']:.4f}")
        print(f"纯 LSTM 准确率:      {results['pure_lstm']['accuracy']:.4f}")
        diff = results['cnn_lstm']['accuracy'] - results['pure_lstm']['accuracy']
        print(f"提升: {diff:.4f} ({diff*100:.1f} 个百分点)")


if __name__ == "__main__":
    main()
