# 路灯状态检测系统

基于YOLOv8的路灯状态检测系统，可以实现对路灯的识别以及路灯状态（亮/不亮/不够亮）的检测。
![results](https://github.com/user-attachments/assets/91d4c446-95d0-4334-afa2-99c35f01f5b0)
![val_batch0_pred](https://github.com/user-attachments/assets/3bff202d-dedf-4662-bca0-f55d82386b49)
![val_batch1_pred](https://github.com/user-attachments/assets/b9e0083a-5540-41ca-8ab1-12f64a954cc7)


## 项目介绍

本项目利用YOLOv8目标检测算法，训练了一个可以识别路灯并判断其工作状态的模型。该模型可以应用于智慧城市建设、市政设施管理等场景，帮助相关部门自动化监控路灯的工作状态，提高城市管理效率。

## 调研范围
本调研报告采用的数据集部分为小组成员亲自拍摄得来，路灯数据来源于清华东路、五道口和明光村三大片区，共七百余张，包括室外路灯、小射灯、枝叶遮挡的路灯，以及亮光、弱光等场景。我们基于实地调研所得路灯的亮度、遮挡状态，对数据集进行了严格的标注，最终数据集分为608张用于训练、72张用于测试和40张用于验证，尽管数据集场景复杂，但最终在识别路灯方面展示出其较稳定的精度，而遮挡判断方面仍待提升，后续考虑增加更多数据集进行拟合收敛
![6110b56514ddc4785fa58b5710aa8fa](https://github.com/user-attachments/assets/2d9b1d03-dca3-4962-9e57-25295b4dde25)
![42abbbb245bf8bce7c02d1b2fa5215b](https://github.com/user-attachments/assets/ab309963-4c60-49a7-972e-1976ba0225de)
![88b45525deebf1095bce5e4838183fa](https://github.com/user-attachments/assets/abb48711-ab8c-4ec2-889f-1d17894781e8)

## 功能特点

- 路灯检测：能够在图像或视频中准确定位路灯位置
- 状态识别：可以判断路灯是否正常工作（亮/不亮/不够亮）
- 高效处理：基于YOLOv8的高效目标检测架构，可以实现实时处理

## 数据集

数据集包含两个类别：
- 0: StreetLamp - 路灯
- 1: Obscured - 遮挡状态

数据集结构：
```
datasets/bvn/
├── images/
│   ├── train/ (训练图像)
│   └── val/ (验证图像)
└── labels/
    ├── train/ (训练标签)
    └── val/ (验证标签)
```

## 模型训练

模型基于YOLOv8n架构训练，使用以下命令进行训练：

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8n-CBAM.yaml')  # 加载预训练的YOLOv8模型
model.train(data="yolo-bvn.yaml", workers=0, epochs=100, batch=16)  # 训练模型
```

## 使用方法

### 安装依赖

```bash
pip install ultralytics
```

### 推理示例

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("./best.pt")

# 进行预测
results = model(source="path/to/image.jpg", save=True, conf=0.30)
```

## 模型文件

- `best.pt`: 训练过程中性能最好的权重文件
- `last.pt`: 训练结束时的最后一个权重文件

## 性能指标

模型在验证集上的性能指标可以在`runs/detect/train10`目录下查看，包括准确率、召回率、混淆矩阵等。

## 应用场景

- 智慧城市路灯监控系统
- 市政设施管理
- 夜间道路安全监测
- 能源使用效率监控

## 未来改进

- 增加更多类别（如路灯损坏、倾斜等状态）
- 优化模型大小，适用于边缘设备
- 开发实时监控系统界面
- 集成到智慧城市管理平台

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
