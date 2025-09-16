# Gen6D 中文使用指南

## 项目概述

Gen6D 是一个先进的6自由度（6DOF）物体姿态估计框架，能够从单张图像中精确估计物体在三维空间中的位置和方向。该项目采用"检测-选择-优化"三阶段流程，结合深度学习与几何优化技术，实现高精度、鲁棒的姿态估计。

## 快速开始

### `run_yolo_version.sh` 脚本使用方法

`run_yolo_version.sh` 是一个便捷的执行脚本，用于运行 Gen6D 的预测功能。该脚本包含了两种不同的预测模式配置。

#### 脚本内容说明

```bash
# 模式1：使用多模型预训练配置（推荐）
python predict.py \
    --video_path data/gs/chair/test/images \
    --output data/gs/chair/test/ \
    --database data/gs/chair/ref/3dgs/point_cloud/iteration_30000/chair.ply \
    --cfg configs/gen6d_multi_pretrain.yaml

# 模式2：使用标准预训练配置（备选）
# python predict.py \
#     --video_path data/gs/chair/test/images/ \
#     --output data/gs/chair/test/ \
#     --database gs/chair \
#     --cfg configs/gen6d_pretrain.yaml
```

### 参数详解

#### `predict.py` 脚本参数

| 参数 | 说明 | 示例值 | 必需 |
|------|------|--------|------|
| `--video_path` | 输入图像序列的目录路径 | `data/gs/chair/test/images` | ✅ |
| `--output` | 输出结果的保存目录 | `data/gs/chair/test/` | ✅ |
| `--database` | 参考数据库路径（3D点云文件或数据库名称） | `data/gs/chair/ref/3dgs/point_cloud/iteration_30000/chair.ply` | ✅ |
| `--cfg` | 配置文件路径 | `configs/gen6d_multi_pretrain.yaml` | ✅ |

### 使用步骤

#### 1. 数据准备

3DGS 点云， 需要用 sh_degree=0进行训练，并且保证点云中心在原点， z 轴方向朝上
#### 2. 运行预测

##### 方法一：直接执行脚本
```bash
# 给脚本添加执行权限
chmod +x run_yolo_version.sh

# 执行脚本
./run_yolo_version.sh
```

##### 方法二：手动执行 Python 命令
```bash
python predict.py \
    --video_path data/gs/chair/test/images \
    --output data/gs/chair/test/ \
    --database data/gs/chair/ref/3dgs/point_cloud/iteration_30000/chair.ply \
    --cfg configs/gen6d_multi_pretrain.yaml
```

#### 3. 查看结果

预测完成后，输出目录将包含以下文件：

```
data/gs/chair/test/
├── images_out/              # 主要输出结果
│   ├── image1-bbox.jpg     # 带3D边界框的可视化结果
│   ├── image1-pose.npy     # 姿态矩阵（4x4变换矩阵）
│   ├── image2-bbox.jpg
│   ├── image2-pose.npy
│   └── ...
└── images_inter/            # 中间结果可视化
    ├── image1_0.jpg        # 中间处理步骤的可视化
    ├── image2_0.jpg
    └── ...
```

### 配置选项

#### 两种预训练配置的区别

1. **`gen6d_multi_pretrain.yaml`**（推荐）
   - 类型：`gen6d_mult`
   - 支持多目标检测
   - 更适合复杂场景

2. **`gen6d_pretrain.yaml`**（标准）
   - 类型：`gen6d`
   - 单目标检测
   - 适用于简单场景

#### 关键配置参数

```yaml
ref_resolution: 128      # 参考图像分辨率
ref_view_num: 64        # 选择阶段使用的视角数量
det_ref_view_num: 32    # 检测阶段使用的视角数量
refine_iter: 3          # 优化迭代次数
```

### 输入数据格式要求

#### 图像要求
- 格式：JPG/PNG
- 尺寸：无严格限制（系统会自动生成相机内参）
- 质量：建议高质量图像以获得更好的结果

#### 3D点云文件
- 格式：PLY 文件
- 内容：物体的3D几何信息
- 获取方式：可通过3D重建技术（如NeRF、3D Gaussian Splatting等）生成

### 故障排除

#### 常见问题

1. **路径不存在错误**
   ```bash
   # 检查输入路径是否正确
   ls data/gs/chair/test/images
   ls data/gs/chair/ref/3dgs/point_cloud/iteration_30000/chair.ply
   ```

2. **内存不足**
   - 减少 `ref_view_num` 和 `det_ref_view_num` 的值
   - 使用较小的输入图像

3. **没有检测到物体**
   - 检查参考数据库是否正确
   - 确认物体在图像中清晰可见
   - 尝试调整配置参数

#### 调试模式

如需查看详细的中间结果，可以检查 `images_inter/` 目录中的可视化文件，这些文件显示了：
- 物体检测结果
- 视角选择过程
- 姿态优化步骤

### 高级用法

#### 自定义数据集

如需使用自己的数据：

1. 按照标准格式组织数据
2. 修改脚本中的路径参数
3. 根据需要调整配置文件

#### 批量处理

可以修改脚本以处理多个对象：

```bash
#!/bin/bash
# 批量处理多个对象
for object in chair table cup; do
    python predict.py \
        --video_path data/gs/${object}/test/images \
        --output data/gs/${object}/test/ \
        --database data/gs/${object}/ref/3dgs/point_cloud/iteration_30000/${object}.ply \
        --cfg configs/gen6d_multi_pretrain.yaml
done
```

## 技术支持

如遇到问题，请：

1. 检查环境依赖是否正确安装
2. 确认数据格式是否符合要求
3. 查看错误日志获取详细信息
4. 参考项目文档中的故障排除章节

---

**注意**：本脚本设计用于演示和测试目的。在生产环境中使用时，请根据具体需求调整参数配置。