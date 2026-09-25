# 📜 经典论文精读专栏

> **状态：🎉 15 篇全部上线，专栏完结**（01–04：Transformer / ResNet / Adam / word2vec；05–15：BERT / GPT-3 / InstructGPT / DPO / LoRA / GAN / DDPM / CLIP / GCN / DQN+PPO / XGBoost）。

> 十五篇塑造了现代 AI 的论文，每篇一页精读：**背景与动机 → 核心思想（人话先行）→ 方法细节（含关键公式推导）→ 实验解读（论文到底证明了什么、没证明什么）→ 后续影响与争议 → 与本库章节的连接**。读原著之前先读这里，读原著之后回来对照。

## 为什么是这十五篇

每一篇都满足至少一条：开创一个范式 / 提出一个至今在用的机制 / 改变了行业的训练方式。按本库章节顺序排列。

| # | 论文（年份） | 对应章节 | 一句话地位 | 状态 |
|---|--------------|----------|-----------|------|
| 01 | Attention Is All You Need（2017） | 第 7 章 | Transformer 开山之作，现代 AI 的架构基座 | ✅ [精读](./01-Attention-Is-All-You-Need.md) |
| 02 | Deep Residual Learning for Image Recognition（2015） | 第 4/5 章 | 残差连接让百层网络可训，"抄近道"的数学 | ✅ [精读](./02-ResNet.md) |
| 03 | Adam: A Method for Stochastic Optimization（2014） | 第 4 章 | 至今默认的优化器，自适应步长 | ✅ [精读](./03-Adam.md) |
| 04 | Efficient Estimation of Word Representations（word2vec, 2013） | 第 7 章 | "看朋友圈识人"的向量思想起点 | ✅ [精读](./04-word2vec.md) |
| 05 | BERT（2018） | 第 7 章 | 双向预训练理解范式的确立 | ✅ [精读](./05-BERT.md) |
| 06 | Language Models are Few-Shot Learners（GPT-3, 2020） | 第 7 章 | 规模出智能的实证里程碑 | ✅ [精读](./06-GPT-3.md) |
| 07 | Training language models to follow instructions（InstructGPT, 2022） | 第 7 章 | RLHF 三阶段流程的出处 | ✅ [精读](./07-InstructGPT.md) |
| 08 | Direct Preference Optimization（DPO, 2023） | 第 7 章 | 跳过奖励模型的偏好优化 | ✅ [精读](./08-DPO.md) |
| 09 | LoRA: Low-Rank Adaptation（2021） | 第 7 章 | 一张卡微调大模型的工程革命 | ✅ [精读](./09-LoRA.md) |
| 10 | Generative Adversarial Networks（GAN, 2014） | 第 12 章 | 对抗训练思想的诞生 | ✅ [精读](./10-GAN.md) |
| 11 | Denoising Diffusion Probabilistic Models（DDPM, 2020） | 第 12 章 | 扩散模型的现代复兴 | ✅ [精读](./11-DDPM.md) |
| 12 | Learning Transferable Visual Models（CLIP, 2021） | 第 12 章 | 图文对齐，多模态与零样本的地基 | ✅ [精读](./12-CLIP.md) |
| 13 | Semi-Supervised Classification with Graph CNN（GCN, 2016） | 第 9 章 | 谱到空域的简化，GNN 实用化起点 | ✅ [精读](./13-GCN.md) |
| 14 | Human-level control through deep RL（DQN, 2015） / Proximal Policy Optimization（PPO, 2017） | 第 10 章 | 深度 RL 的两次关键跨越（合读一页） | ✅ [精读](./14-DQN-PPO.md) |
| 15 | XGBoost: A Scalable Tree Boosting System（2016） | 第 3 章 | 表格数据至今仍在的王座 | ✅ [精读](./15-XGBoost.md) |

## 精读页统一结构

```
📌 论文速览（一张表：谁/何时/解决什么/影响因子级别的引用地位）
🌱 一句话核心思想（先给直觉）
🧠 方法精读（关键公式的完整推导——含论文原文记号到本书记号的翻译）
📊 实验解读（证明了什么 / 没证明什么 / 消融怎么读）
🔁 后世影响（谁站在它肩上；争议与反思）
🔗 在本库中的位置（对应章节页链接）
```

## 写作红线

- 公式推导**自己重写**（含记号翻译），不逐句翻译论文；原文图表不搬运，用自绘示意
- "没证明什么"与"证明了什么"同样重要——防止把论文读成神话
- 每页 400~600 行，公式后必须有人话
