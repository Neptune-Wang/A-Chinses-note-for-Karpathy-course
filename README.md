# Zero to Hero 中文学习笔记

这是我学习 Andrej Karpathy「Neural Networks: Zero to Hero」课程时整理的笔记，记录从 Micrograd、Makemore 到 GPT 和 Tokenizer 的学习过程。

笔记以中文解释为主，穿插英文术语、代码片段、推导和配图，也包含学习过程中补充的 Python 基础知识。目前整理了 8 讲，可以按下面的目录依次阅读。

## 笔记目录

| 讲次 | 笔记 | 主要内容 |
| --- | --- | --- |
| 1 | [Micrograd](Lecture1_MicroGrad/Lecture1_MicroGrad.md) | Python 基础、自动微分、反向传播与多层感知机 |
| 2 | [Makemore：Bigram](Lecture2_Building%20Makemore/Lecture2_Building%20Makemore.md) | 字符级语言模型、计数方法与神经网络方法 |
| 3 | [Makemore：MLP](Lecture3_%20Makemore%202%20%28MLP%29/Lecture3_%20Makemore%202%20%28MLP%29.md) | 嵌入层、多层感知机、学习率与数据集划分 |
| 4 | [Activations、Gradients 与 BatchNorm](Lecture4_Makemore3%EF%BC%88Activations%2CGradients%2CBatchNorm%EF%BC%89/Lecture4_Makemore3%EF%BC%88Activations%2CGradients%2CBatchNorm%EF%BC%89.md) | 初始化、激活值与梯度分布、批量归一化 |
| 5 | [手动反向传播](Lecture5/Lecture5.md) | 逐步推导梯度，并与 PyTorch 自动微分比较 |
| 6 | [Makemore：WaveNet](Lecture%206%EF%BC%9AMakemore5%EF%BC%88Wavenet%EF%BC%89/Lecture%206%EF%BC%9AMakemore5%EF%BC%88Wavenet%EF%BC%89.md) | 分层组合上下文、加深网络与批量归一化细节 |
| 7 | [从零实现 GPT](Lecture%207_%20GPT1%EF%BC%88Scratch%EF%BC%89/Lecture%207_%20GPT1%EF%BC%88Scratch%EF%BC%89.md) | 自注意力、Transformer 结构与文本生成 |
| 8 | [GPT Tokenizer](Lecture8_%20GPT2%28Tokenization%29/Lecture8_%20GPT2%28Tokenization%29.md) | Unicode、字节对编码（BPE）、编码与解码 |

## 阅读方式

- 在 GitHub 上点击目录中的笔记，即可阅读正文、代码和配图。
- 下载到本地后，可用 Obsidian 或支持 Markdown 的编辑器打开。
- 每讲的图片保存在同级的「图片和附件」目录中。移动或分享笔记时，请保留这一文件夹结构。
- 代码片段的运行环境和数据准备，请结合对应笔记与原课程说明。

## 课程来源

- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html)
- [原课程 GitHub 仓库：karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

这里记录的是我的学习理解与整理；课程讲解和示例代码请参考以上原始资料。
