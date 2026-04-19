# MediaPipe Fitness Action Recognition
# 毕设自用
基于 MediaPipe 与轻量级深度学习模型的健身动作智能识别系统。项目提供动作识别、实时计数、动作评分、训练记录管理与 AI 训练报告能力，适合作为毕业设计展示和二次开发基础。

## 功能概览

- 基于 MediaPipe Pose Landmarker 提取 17 个核心关键点
- 支持 `squat`、`push-up`、`crunches`、`lunge` 四类动作识别
- 提供实时动作计数、姿态评分与纠错提示
- 使用 Streamlit 构建前端交互界面
- 使用 SQLite 存储用户、训练记录、计划与 AI 配置
- 包含关键点提取、模型训练、模型评估完整脚本

## 模型结果

当前仓库附带的 `1D-CNN + LSTM` 模型在测试集上的结果如下：

- Accuracy: `0.9188`
- Macro F1: `0.9205`

详细指标见 `models/evaluation_results.json`。

## 项目结构

```text
.
├─app.py                  # Streamlit 主应用
├─config.py               # 全局配置
├─database.py             # SQLite 数据层
├─extract_keypoints.py    # 视频关键点提取与样本生成
├─train.py                # 模型训练脚本
├─evaluate.py             # 模型评估脚本
├─inference.py            # 实时推理引擎
├─models/                 # 训练结果、评估图、模型权重
└─data/dataset_stats.json # 数据集统计摘要
```

## 环境要求

- Python 3.10+
- Windows 摄像头环境更容易直接运行当前界面

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

首次启动会在本地创建 SQLite 数据库，并自动生成默认账号：

- 管理员：`admin / admin123`
- 普通用户：`user / user123`

## 模型与素材说明

仓库默认不包含原始训练/测试视频、提取后的 `.npy` 特征文件、运行日志与本地数据库。

训练或重新提取关键点时，需要准备数据集目录。程序按以下优先级查找素材目录：

1. 环境变量 `FITNESS_MATERIAL_DIR`
2. 仓库内 `material/`
3. 兼容旧工程结构的上级目录 `../material/`

推荐的数据集目录结构：

```text
material/
├─train/
│ ├─squat/
│ ├─push-up/
│ ├─crunches/
│ └─lunge/
├─verify/
└─test/
```

另外，运行姿态检测前还需要将 MediaPipe 的 `pose_landmarker_heavy.task` 放到 `models/` 目录。该文件为第三方模型资源，未随本仓库一起分发。

## 常用命令

```powershell
python extract_keypoints.py
python train.py
python evaluate.py
streamlit run app.py
```

## 开源范围

当前仓库保留：

- 核心源代码
- 训练好的模型权重与评估结果
- 项目依赖与运行说明

当前仓库排除：

- 原始视频素材
- 提取后的大体积特征文件
- 本地数据库与用户记录
- 本地日志、缓存和私有配置
- 第三方 `pose_landmarker_heavy.task` 模型文件

## License

This project is released under the MIT License.
