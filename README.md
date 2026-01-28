# Qwen2.5 模型知识边界探测工具

## 📋 项目说明

本项目提供了一套完整的工具，用于探测 Qwen2.5 系列模型的知识边界。通过多种提示词变体测试模型对特定知识的掌握程度，将知识点分类为：
- **Known（已知）**: 模型对所有提示词变体都能正确回答
- **Unknown（未知）**: 模型对所有提示词变体都无法正确回答
- **Ambiguous（模糊）**: 模型对部分提示词能正确回答，部分不能

## 🚀 主要改进

### 原始代码的问题修复：
1. **Prompt格式化问题**: 改进了 `format_prompt_with_subject` 函数，支持多种占位符格式
2. **内存管理**: 添加了自动内存清理和OOM错误处理
3. **数据持久化**: 添加了实时flush，避免意外中断时数据丢失
4. **错误处理**: 增强了异常处理机制

### 新增功能：
1. **分布式支持**: 支持多GPU和多节点并行处理
2. **自适应批处理**: 根据模型大小自动调整batch size
3. **断点恢复**: 支持从中断点继续运行
4. **结果合并**: 自动合并分布式运行的结果
5. **智能资源分配**: 根据模型大小自动配置计算资源

## 📁 文件说明

```
.
├── probe_model_distributed.py    # 主要探测脚本（分布式版本）
├── merge_distributed_results.py  # 结果合并脚本
├── submit_probe_jobs.sh          # 批量提交集群任务脚本
├── test_single_model.sh          # 单模型测试脚本
└── README.md                      # 本文档
```

## 🔧 使用方法

### 1. 准备工作

首先，将所有脚本上传到集群：

```bash
# 上传脚本到集群存储
scp probe_model_distributed.py user@cluster:/mnt/afs/your_path/
scp merge_distributed_results.py user@cluster:/mnt/afs/your_path/
scp submit_probe_jobs.sh user@cluster:/mnt/afs/your_path/
scp test_single_model.sh user@cluster:/mnt/afs/your_path/
```

### 2. 配置路径

编辑脚本中的路径配置：

```bash
# 在 submit_probe_jobs.sh 中修改：
INPUT_FILE="/mnt/afs/path/to/your/counterfact.jsonl"  # 数据集路径
BASE_OUTPUT_DIR="/mnt/afs/probe_results"              # 输出目录
SCRIPT_PATH="/mnt/afs/probe_model_distributed.py"     # 脚本路径

# 修改模型路径
model_paths=(
    ["qwen2.5-0.5B-Instruct"]="/mnt/afs/models/Qwen2.5-0.5B-Instruct"
    # ... 其他模型路径
)
```

### 3. 测试运行

先用小模型测试确保环境正确：

```bash
# 编辑 test_single_model.sh 中的路径
vim test_single_model.sh

# 运行测试
bash test_single_model.sh

# 查看测试结果
sco acp jobs logs --workspace-name <workspace> -j probe_test_qwen2.5-0.5B-Instruct
```

### 4. 正式运行

```bash
# 运行批量探测
bash submit_probe_jobs.sh

# 选择要运行的模型：
# 1) 全部模型
# 2) 小模型 (0.5B, 1.5B, 3B)
# 3) 中等模型 (7B, 14B)
# 4) 大模型 (32B, 72B)
# 5) 自定义选择
```

### 5. 监控任务

```bash
# 查看所有任务状态
sco acp jobs list --workspace-name <workspace>

# 查看特定任务日志
sco acp jobs logs --workspace-name <workspace> -j <jobname>

# 取消任务
sco acp jobs cancel --workspace-name <workspace> -j <jobname>
```

## 📊 资源配置说明

| 模型 | 节点数 | GPU/节点 | 总GPU数 | 预估时间* |
|------|--------|----------|---------|-----------|
| 0.5B | 1 | 1 | 1 | ~2小时 |
| 1.5B | 1 | 1 | 1 | ~3小时 |
| 3B | 1 | 2 | 2 | ~4小时 |
| 7B | 1 | 4 | 4 | ~5小时 |
| 14B | 2 | 4 | 8 | ~6小时 |
| 32B | 2 | 8 | 16 | ~8小时 |
| 72B | 4 | 8 | 32 | ~10小时 |

*预估时间基于2万样本数据集

## 🎯 参数说明

### probe_model_distributed.py 参数

- `--model_name`: 模型名称（如 qwen2.5-7B-Instruct）
- `--model_path`: 模型权重路径
- `--input_file`: 输入JSONL文件路径
- `--output_dir`: 输出目录
- `--f1_threshold`: F1分数阈值（默认0.5）
- `--batch_size`: 批处理大小（默认自动调整）
- `--max_new_tokens`: 生成的最大token数（默认32）
- `--test_run_limit`: 测试模式下的样本数限制
- `--resume`: 是否从检查点恢复
- `--checkpoint_interval`: 保存检查点的间隔（默认100）

## 📈 输出文件说明

运行完成后，会在输出目录生成以下文件：

```
output_dir/
├── known_bucket.jsonl              # 已知类别的原始数据
├── known_bucket_details.jsonl      # 已知类别的详细探测结果
├── unknown_bucket.jsonl            # 未知类别的原始数据
├── unknown_bucket_details.jsonl    # 未知类别的详细探测结果
├── ambiguous_bucket.jsonl          # 模糊类别的原始数据
├── ambiguous_bucket_details.jsonl  # 模糊类别的详细探测结果
└── summary_report.txt              # 汇总统计报告
```

## ⚠️ 注意事项

1. **模型路径**: 确保所有模型路径正确，模型文件已下载完整
2. **存储空间**: 确保输出目录有足够的存储空间（建议预留100GB）
3. **内存使用**: 72B模型需要特别注意内存配置，可能需要调整 `max_memory` 参数
4. **网络配置**: 分布式运行需要确保节点间网络通信正常
5. **数据格式**: 输入JSONL文件必须符合示例格式

## 🔍 故障排查

### 常见问题及解决方案：

1. **CUDA Out of Memory**
   - 减小 batch_size
   - 检查是否有其他进程占用GPU
   - 对大模型启用量化

2. **分布式通信失败**
   - 检查NCCL环境变量设置
   - 确认节点间网络连通性
   - 验证端口（29500）未被占用

3. **模型加载失败**
   - 验证模型路径是否正确
   - 检查模型文件完整性
   - 确认有足够的内存/显存

4. **结果不一致**
   - 确保使用相同的随机种子
   - 检查tokenizer配置是否一致
   - 验证prompt格式化是否正确

## 📝 结果分析

探测完成后，可以通过以下方式分析结果：

```python
import json
import pandas as pd

# 读取汇总报告
with open("output_dir/summary_report.txt", "r") as f:
    print(f.read())

# 分析详细结果
known_details = []
with open("output_dir/known_bucket_details.jsonl", "r") as f:
    for line in f:
        known_details.append(json.loads(line))

# 转换为DataFrame进行进一步分析
df = pd.DataFrame(known_details)
print(f"平均F1分数: {df['probe_details'].apply(lambda x: sum(d['f1_score'] for d in x)/len(x)).mean()}")
```

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可

本项目仅供研究使用。