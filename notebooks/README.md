# 📓 实战 Notebook

> 每章一个可运行的 Jupyter notebook，把正文的"动手试试"扩展成完整实验：代码 + 逐段中文讲解 + 图表 + 改参数观察。克隆仓库后 `jupyter lab` 打开即可复现。

## 使用

```bash
pip install -r requirements.txt   # 或按需安装：numpy / scikit-learn / matplotlib / torch / statsmodels
jupyter lab
```

约定：数据全部**代码生成或内置数据集**（不依赖外部下载，保证离线可跑）；每个 notebook 从上到下完整执行一遍即对应正文实验；随机种子固定，结果可复现。

## Notebook 清单（随章节逐步上线）

| Notebook | 对应章节 | 内容 | 状态 |
|----------|----------|------|------|
| ch02-数学基础.ipynb | 第 2 章 | 点积相似度 / 手算梯度验证 / 贝叶斯更新 | ✅ |
| ch03-机器学习.ipynb | 第 3 章 | 回归分类全流程 / 交叉验证 / 集成对比 / 学习曲线诊断 | ✅ |
| ch04-深度学习基础.ipynb | 第 4 章 | 前向/反向传播手算对照 / 激活函数梯度现场 / ReLU 分段线性 / 优化器对比 / 学习率三档 / 训练循环模板（torch） | ✅ |
| ch05-计算机视觉.ipynb | 第 5 章 | 卷积手算与形状流 / 卷积核动物园 / 池化与感受野 / 残差梯度证据 / CNN 训手写数字（torch + sklearn 离线替身） | ✅ |
| ch06-时间序列.ipynb | 第 6 章 | 分解 / ARIMA 定阶与回测 / LSTM 外推 | ⏳ |
| ch07-NLP与LLM.ipynb | 第 7 章 | 注意力手写 / 小模型生成 / 最小 RAG | ⏳ |
| ch08-推荐系统.ipynb | 第 8 章 | 协同过滤 / 双塔训练与 ANN 检索 / CTR 排序 | ⏳ |
| ch09-图神经网络.ipynb | 第 9 章 | 图统计 / 消息传递手推 / PyG Cora | ⏳ |
| ch10-强化学习.ipynb | 第 10 章 | Q-Learning CliffWalking / DQN CartPole | ⏳ |
| ch11-因果推断.ipynb | 第 11 章 | 混淆模拟 / DID / PSM / uplift | ⏳ |
| ch12-生成模型.ipynb | 第 12 章 | 最小 GAN / 玩具扩散训练与采样 | ⏳ |

## 写作规范（贡献者适用）

1. 每个 notebook 结构：标题与对应章节链接 → 环境准备 → 逐实验小节（每节先 markdown 讲目标，再代码，后图表与结论）→ 收尾"改参数建议"清单
2. 单元即段落：一个代码单元只做一件事；图表必须带中文标题与轴标签
3. 与正文页面互相链接：notebook 首页链到对应章 README，正文页"动手试试"链接到 notebook
