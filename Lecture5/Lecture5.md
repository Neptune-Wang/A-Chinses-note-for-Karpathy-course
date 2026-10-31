# Introduction and Praperation

  
在前两讲中，我们已经构建了一个相当可靠的多层感知机（MLP），用于生成更多姓名。我们已经知道 MLP 的结构和工作方式。训练中的一个关键步骤是调用 `loss.backward()`：它会触发 PyTorch 的反向传播程序，计算梯度，最终用来改进模型参数。**但我们还没有真正研究过这个过程究竟如何完成。**

此前我们通过 Micrograd 简要研究过反向传播，但当时处理的_只有__标量_。 为了真正理解反向传播，现在必须把相关思想从标量扩展到张量。最好的学习方法是：使用 PyTorch，从零开始为基于张量的 Makemore MLP 手动实现一次完整的反向传播。

我们先进行一些固定的初始化操作：

```Python
# 前几个单元格与上一讲相比没有变化
import torch
 import random
 import torch.nn.functional as F
 import matplotlib.pyplot as plt
 %matplotlib inline
```

```Python
# 读入所有单词
words = open('names.txt', 'r').read().splitlines()
# .splitlines()的作用是按照换行符把字符串拆分成多行，并返回一个列表
print('Samples:', words[:8])
print('Size:', len(words))
print('Largest:', max(len(w) for w in words))
```

我们仍然按照此前的方式构造数据集：`X` 中的输入是字符三元上下文，`Y` 中的目标是紧随其后的下一个字符。 网络结构也与之前相同：一个嵌入层、一个线性层、批归一化层、一个非线性激活函数，以及最后一个用于生成下一个字符预测 logits 的线性层。 损失函数仍然使用交叉熵损失。

```Python
构建数据集
block_size = 3 # 上下文长度：用于预测下一个字符的字符数量
def build_dataset(words):
 X, Y = [], []
  for w in words:
 context = [0] * block_size
 for ch in w + '.':
 ix = stoi[ch]
 X.append(context)
 Y.append(ix)
 context = context[1:] + [ix] # 删除最前面的字符并在末尾追加新字符
  X = torch.tensor(X)
 Y = torch.tensor(Y)
 print(X.shape, Y.shape)
 return X, Y
random.seed(42)
打乱数据，并划分为三个子集
random.shuffle(words)
 n1 = int(0.8len(words))
 n2 = int(0.9len(words))
Xtr,  Ytr  = build_dataset(words[:n1])   # 80% 训练集
 Xdev, Ydev = build_dataset(words[n1:n2]) # 10% 验证集
 Xte,  Yte  = build_dataset(words[n2:])   # 10% 测试集
```

**准备工作已经完成，下面正式开始。** 为了学习如何计算梯度，我们会定义一个新的 `cmp()` 函数，用它把手工计算的梯度与 PyTorch 自动计算的梯度进行比较：

```Python
# 稍后会使用的辅助函数
# 用于比较手工计算的梯度与 PyTorch 计算的梯度

# s 字符串，用来说明比较的是哪个张量
# dt 是手工计算的梯度
# t 是由 PyTorch 计算梯度的张量
def cmp(s,dt,t):
    ex = torch.all(dt == tr.grad).item()  # 完全一致
    app = torch.allclose(dt, t.grad)    # 足够接近（近似一致）
    maxdiff = (dt - t.grad).abs().max().item()  # 二者的最大差异，理想情况下应为零
```

其中，`torch.all()`函数用于比较两个张量是否完全相等，输出 Bool 值，`torch.allclose()`则容许两个变量之间有一定误差，具体规则如下：

> 判断标准：$$|a-b|\leq atol+rtol|b$$
> 
> `atol`：绝对误差阈值，默认 1e-8
> 
> `rtol`：相对误差阈值，默认 1e-5。

  

下面按照之前的方式搭建 MLP。请注意，很多参数（例如偏置）的初始化方式并不标准。这是因为，把参数全部初始化为零等做法有时会_掩盖_反向传播实现中的错误。通过加入 `0.1` 的缩放因子，使参数具有一定的随机变化，错误实现就更容易暴露出来。

```Python
n_embd = 10   # 神经网络字符嵌入向量的维数
n_hidden = 64 # MLP 隐藏层的神经元数量

g = torch.Generator().manual_seed(2147483647)       # 保证结果可复现
C  = torch.randn((vocab_size, n_embd), generator=g) # 字符嵌入表

prop_scale = 0.1 # 加入缩放因子，避免掩盖反向传播中的潜在错误

# 第 1 层
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) # (3x10)x64
W1 *= (5/3)/((n_embd * block_size)**0.5)              # 初始权重缩放
b1 = torch.randn(n_hidden, generator=g) * prop_scale  # 这里保留 64 个 b1 偏置以便分析，实际上并非必需

# 批归一化
bngain = torch.randn((1, n_hidden))*prop_scale + 1.0
bnbias = torch.randn((1, n_hidden))*prop_scale

# 第 2 层
W2 = torch.randn((n_hidden, vocab_size), generator=g) * prop_scale # 64x27
b2 = torch.randn(vocab_size, generator=g) * prop_scale # 与 b1 不同，这些偏置有助于暴露梯度计算中的细微错误

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # 参数总数

for p in parameters:
  p.requires_grad = True
```

正常情况下使用批归一化就不需要偏置项`b1`,但这里保留以便后续分析。权重矩阵和偏置向量被初始化为较小但具有一定随机变化的数值。 与之前一样，我们使用批量输入，批量大小设为 32。下面只构造其中一个批次。只要让一个批次完成一次前向传播，就足以完成本节的分析：

```Python
batch_size = 32
n = batch_size  # 为了方便，用 n 作为批次大小的简写
# 构造mini batch
ix = torch.randint(0, Xtr.shape[0], (batch_size,),generator = g)    # Xtr的形状为 182625*3
Xb, Yb = Xtr[ix], Ytr[ix]
```

为了能够逐步手工反向传播，下面将前向传播过程，尤其是 BatchNorm 与交叉熵损失，改写成 PyTorch 内部实现的“分步版本”。这样，我们就可以依次对每个中间步骤执行反向传播：

```Python
# forward pass
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1) # 拼接向量
# 第一个线性层
hprebn = embcat @ W1 +b1
# BatchNorm层
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1 / (n-1) * (bndiff2).sum(0,keepdim=True)   # 这里使用 Bessel's correction，分母是 n-1
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv  # 这两步进行BN
hpreact = bngain * bnraw + bnbias
# 非线性激活
h = torch.tanh(hpreact) # 隐藏层

# 第二个线性层
logits = h @ W2 + b2 # 隐藏层
# 交叉熵损失（此前这些步骤都包含在 F.cross_entropy(logits, Yb) 中）
logits_maxes = logits.max(1, keepdim = True).values
norm_logits = logits - logits_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1 # 如果改用 1.0 / counts_sum，反向传播结果无法做到逐位完全一致
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
# Pytorch 反向传播
for p in parameters:
    p.grad = None   # 去除上一次反向传播残余的梯度
    # 保留中间张量的梯度，反向传播结束后不会被释放，便于比较
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,  norm_logits, logits_maxes, logits, h, hpreact, bnraw,
bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani, embcat, emb]:
    t.retain_grad()
loss.backward()
loss
```

输出结果如下：

```Plain
tensor(3.3641, grad_fn=<NegBackward0>)
```

现在我们已经拥有由 PyTorch 计算出的梯度，可以把它们作为参照。下一步就是自行实现反向传播，并逐项比较结果。

# Excersise1：Backward Pass

> 回顾 Micrograd 中的结论： 加法 c = a + b 的梯度： a.grad = c.grad * (1) b.grad = c.grad * (1) 乘法 c = a * b 的梯度： a.grad = c.grad * b.data b.grad = c.grad * a.data

## 1. logprobs

`loss = -logprobs[range(n), Yb].mean()`

**梯度的形状与被求导变量相同**，因此 `dlogprobs` 和 `logprobs` 一样，形状必须是 `(32, 27)`。

`logprobs[range(n), Yb]`的形状是（32,）推导如下：

> `range(n)`为`[0, 1, ..., 31]`，不妨设`Yb`为`[5,2,8,...,1]`,
> 
> PyTorch 会一一配对两个索引：
> 
> ```Python
> logprobs[
>     [0,1,2,...,31],
>     [5,2,8,...,1]
> ]
> ```
> 
> 等价于：
> 
> ```Python
> [
>  logprobs[0,5],
>  logprobs[1,2],
>  logprobs[2,8],
>  ...
>  logprobs[31,1]
> ]
> ```
> 
> 而`logprobs[0,5]`表示`logprobs`中的 (0,5)元。

这意味着只有`logprobs`中被选中的 32 个元素才会影响梯度。**`logprobs`** **中所有未被** **`[range(n), Yb]`** **选中的值都不会影响** **`loss`****，因此它们的梯度为零。**

三个数 a,b,c 的负均值为
$$-\frac{a+b+c}{3}=(-\frac13)a+(-\frac13)b+(-\frac13)c$$
因此可以得到
$$\frac{\partial(-\operatorname{mean})}{\partial a}=\frac{\partial(-\operatorname{mean})}{\partial b}=\frac{\partial(-\operatorname{mean})}{\partial c}=-\frac13。$$
$$\frac{\partial\,\text{loss}}{\partial\,\text{logprobs}} =\underline{\underline{-\frac1n}}$$
**因此，`logprobs` 中所有被选中的元素，其梯度都是 1/n；这里的 n 是 `range(n)` 中的行数32。

```Python
# LOGPROBS
dlogprobs = torch.zeros_like(logprobs) # 生成一个和 logprobs 形状相同的全0 张量
dlogprobs[range(n), Yb] = -1.0 / n     
cmp('logprobs', dlogprobs, logprobs)
```
## 2. probs

`probs` 直接且仅用于通过 `logprobs = probs.log()` 计算 `logprobs`，这里的对数是 $\ln(\text{probs})$。 此处必须小心，不能漏掉链式法则。

由于 `probs` 会影响 `logprobs`，相应地，梯度 `dlogprobs` 会通过链式法则影响 `dprobs`：

$$\text{dprobs} = \frac{\partial\log_e(\text{probs})}{\partial\text{probs}} \cdot \text{dlogprobs}$$

还可以检查 `logprobs` 和 `probs` 的形状：二者都是 `(32, 27)`，因此这里没有额外的形状问题。$\ln(\text{probs})$对 $\text{probs}$ 的导数为：
$$
\frac{\partial\ln(\text{probs})}{\partial\text{probs}} = \frac{1}{\text{probs}}
$$
再与 `dlogprobs` 相乘，可得：
$$
d\text{probs} = \frac{1}{\text{probs}}\cdot d\text{logprobs}
$$
代码实现如下：

```Python
# PROBS
dprobs = (1 / probs) * dlogprobs # (32x27), elementwise multiplication
cmp('probs', dprobs, probs)
```

输出结果：
logprobs | exact: True | approximate: True | maxdiff: 0.0

## 3. counts_sum_inv

`counts_sum_inv` 只在一个地方被使用：`probs = counts * counts_sum_inv`。只有 `probs` 依赖 `counts_sum_inv`，但在计算 `probs` 时，`counts_sum_inv` 与 `counts` 共同参与运算，因此二者的梯度彼此相关。

`counts` 是一个 `(32, 27)` 矩阵；`counts_sum_inv` 是一个 `(32, 1)` 向量。<br>
执行乘法时，PyTorch 会沿 `counts` 的列方向广播 `counts_sum_inv`，使其等效为一个 `(32, 27)` 矩阵。利用链式法则，可以分别处理导致该结果的两种操作：乘法与广播。

对于乘法部分：

$$
\text{grad}\left(\text{counts\_sum\_inv}\right)=\frac{\partial(\text{counts}\cdot\text{counts\_sum\_inv})}
{\partial\text{counts\_sum\_inv}}\cdot \text{dprobs}
=
\text{counts}\cdot\text{dprobs}
$$

其中，乘上 `dprobs` 来自链式法则。

对于广播部分：
在 Lecture1 中我们了解到：如果一个值在多个位置被重复使用，那么来自这些使用位置的梯度需要相加。

>考虑一个简单例子：
  前向： $y=x+x+x$ ,也可以看成：$y=3x$
>如果： $\dfrac{\partial L}{\partial y}=1$
>那么：$\dfrac{\partial L}{\partial x} = 1+1+1=3$】

前向：
$$ probs_{ij} = counts_{ij} \times counts\_sum\_inv_i $$
注意，对于固定第 $i$ 个样本：$counts\_sum\_inv_i$被复制到了 27 个类别：
$$ \begin{aligned} probs_{i0}&=counts_{i0}c_i\\ probs_{i1}&=counts_{i1}c_i\\ ...\\ probs_{i26}&=counts_{i26}c_i \end{aligned} $$
其中：
$$ c_i=counts\_sum\_inv_i $$
因此每一个类别都会给$\dfrac{\partial Loss}{\partial c_i}$贡献一部分。
根据乘法求导：
$$ \frac{\partial probs_{ij}} {\partial c_i} = counts_{ij} $$
所以：
$$ \frac{\partial L}{\partial c_i} = \sum_j counts_{ij} \frac{\partial L}{\partial probs_{ij}} $$
写成矩阵形式：
$$ \boxed{ \frac{\partial probs} {\partial counts\_sum\_inv} = \sum_i (counts\cdot dprobs)_{:,i} } $$
这里的求和就是：`counts_sum_inv` 被广播到 `counts` 的全部列上，因此它的梯度应当把第一步的结果沿列方向求和。

代码实现如下：

```python
# COUNTS_SUM_INV

dcounts_sum_inv = (counts * dprobs).sum(1,keepdim=True)

cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
```

输出结果：
probs | exact: True | approximate: True | maxdiff: 0.0

## 4. counts_sum

`counts_sum` 仅用于通过 `counts_sum_inv = counts_sum**-1` 计算 `counts_sum_inv`。<br>
运行代码如下：

```python
# COUNTS_SUM
dcounts_sum = -counts_sum**(-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)
```

## 5. counts

有两条路径依赖于 `counts`：
- `probs`：`probs = counts * counts_sum_inv`
- `counts_sum_inv`：先通过 `counts_sum = counts.sum(1, keepdims=True)`，再计算 `counts_sum_inv = counts_sum**-1`
也就是说，`counts` 这个节点被使用了两次：一次用于 `probs`，一次用于 `counts_sum_inv`。

`counts` 的形状为 `(32, 27)`，`counts_sum_inv` 的形状为 `(32, 1)`。
 $\dfrac{\partial probs}{\partial counts}=counts\_ sum\_ inv$，结合链式法则可以求得第一部分梯度。

现在处理 `counts` 的第二条使用路径：

```python
counts_sum = counts.sum(1, keepdims=True)
```

- `counts` 的形状是 `(32, 27)`
- `counts_sum` 的形状是 `(32, 1)`

可以把这个运算写成：

$$
\begin{bmatrix}
a_{11}+a_{12}+a_{13}\\
a_{21}+a_{22}+a_{23}\\
a_{31}+a_{32}+a_{33}
\end{bmatrix}
=
\begin{bmatrix}
b_1\\
b_2\\
b_3
\end{bmatrix}
$$

这些 `b`（即 `counts_sum` 向量）如何依赖于 `a`（即 `counts` 矩阵）？<br>
以 $b_1$ 为例，它只依赖 $a_{11}$、$a_{12}$ 和 $a_{13}$；相对于 $b_1$，其他所有 $a$ 的局部导数都是零。而 $a_{11}$、$a_{12}$、$a_{13}$ 对应的局部导数都是 $1$，例如
$\frac{\partial b_1}{\partial a_{11}}=1$。

在链式法则中，需要用局部导数乘以上游梯度。由于局部导数都是 $1$，因此每个 $b_i$ 的梯度会不加缩放地流向参与计算该 $b_i$ 的所有 $a$。<br>
运行代码如下：
```python
# COUNTS

# 第 1 部分：

dcounts = counts_sum_inv * dprobs

  

# 第 2 部分：

dcounts += torch.ones_like(counts) * dcounts_sum

  

cmp('counts', dcounts, counts)
```

## 6. norm_logits

`norm_logits` 仅用于通过 `counts = norm_logits.exp()` 计算 `counts`。<br>
$e^x$ 的局部导数仍然是 $e^x$。由于代码中已经有 `counts = norm_logits.exp()`，因此可以直接把 `counts` 作为局部导数：

```python
dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)
```

## 7. logits_maxes

`logits` 和 `logit_maxes` 通过

```python
norm_logits = logits - logit_maxes
```

共同计算 `norm_logits`。`norm_logits` 与 `logits` 的形状都是 `(32, 27)`，而 `logit_maxes` 的形状为 `(32, 1)`。

这个运算基本相当于：

$$
\begin{bmatrix}
c_{11}&c_{12}&c_{13}\\
c_{21}&c_{22}&c_{23}\\
c_{31}&c_{32}&c_{33}
\end{bmatrix}
=
\begin{bmatrix}
a_{11}&a_{12}&a_{13}\\
a_{21}&a_{22}&a_{23}\\
a_{31}&a_{32}&a_{33}
\end{bmatrix}
-
\begin{bmatrix}
b_1&b_1&b_1\\
b_2&b_2&b_2\\
b_3&b_3&b_3
\end{bmatrix}
$$

例如：

$$
c_{32}=a_{32}-b_3
$$

需要考察每个 `c` 是如何得到的。<br>
每个 `c` 都是某个 `a` 与某个 `-b` 的和。<br>
每个 `c` 对相应 `a` 的导数为 $1$，对相应 `b` 的导数为 $-1$。

换句话说，每个 `c` 中的上游梯度会原样流向对应的 `a`，并以负号流向对应的 `b`。<br>
因此有 $dc=da$，也就是：
$$
\text{dlogits}=\text{dnorm\_logits}
$$

同时：
$$
\text{dlogit\_maxes}
=
-\operatorname{sum}(\text{dnorm\_logits})
$$
此时 `dlogit_maxes` 已经完整，而 `dlogits` 只完成了一部分。

```python
dlogits = dnorm_logits.clone()
dlogit_maxes = -1 * dnorm_logits.sum(1, keepdim=True)

cmp('logit_maxes', dlogit_maxes, logit_maxes)
```

## 8. Logits

除了参与 `norm_logits = logits - logit_maxes` 之外，形状为 `(32, 27)` 的 `logits` 还用于：

```python
logit_maxes = logits.max(1, keepdim=True).values
```

调用 `.max(1)` 时，PyTorch 不仅可以得到每一行的最大值，也能得到最大值在 `logits` 中对应的索引，这对于反向传播十分方便。

在每个被取出的最大值位置，局部导数应为 $1$；其他位置为 $0$。<br>
换句话说，我们需要按照最大值所在索引，把 `dlogit_maxes` 散射回 `dlogits`，并将这些值加到对应位置。

```python
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
cmp('logits', dlogits, logits)
```

## 9. h, W2 and b2:the Linear Layer

`h` 仅用于通过 `logits = h @ W2 + b2` 计算 `logits`。

- `dlogits` 的形状是 `(32, 27)`
- `h` 的形状是 `(32, 64)`
- `W2` 的形状是 `(64, 27)`
- `b2` 的形状是 `(27)`

为了理解这里发生了什么，先写出一个较小的具体例子：

$$
\begin{bmatrix}
d_{11}&d_{12}\\
d_{21}&d_{22}
\end{bmatrix}
=
\begin{bmatrix}
a_{11}&a_{12}\\
a_{21}&a_{22}
\end{bmatrix}
\begin{bmatrix}
b_{11}&b_{12}\\
b_{21}&b_{22}
\end{bmatrix}
+
\begin{bmatrix}
c_1&c_2\\
c_1&c_2
\end{bmatrix}
$$

等价地：
$$
\begin{aligned}
d_{11}&=a_{11}b_{11}+a_{12}b_{21}+c_1\\
d_{12}&=a_{11}b_{12}+a_{12}b_{22}+c_2\\
d_{21}&=a_{21}b_{11}+a_{22}b_{21}+c_1\\
d_{22}&=a_{21}b_{12}+a_{22}b_{22}+c_2
\end{aligned}
$$
由 `dlogits`，我们已经知道$\dfrac{\partial L}{\partial d_{11}}$、$\dfrac{\partial L}{\partial d_{12}}$、$\dfrac{\partial L}{\partial d_{21}}$ 和$\dfrac{\partial L}{\partial d_{22}}$。<br>
现在希望计算 $L$ 对 `h`，也就是例子中的 $a$ 的偏导数：
$$
\frac{\partial L}{\partial a_{11}}
=
\frac{\partial L}{\partial d_{11}}b_{11}
+
\frac{\partial L}{\partial d_{12}}b_{12}
$$
$$
\frac{\partial L}{\partial a_{12}}
=
\frac{\partial L}{\partial d_{11}}b_{21}
+
\frac{\partial L}{\partial d_{12}}b_{22}
$$
$$
\frac{\partial L}{\partial a_{21}}
=
\frac{\partial L}{\partial d_{21}}b_{11}
+
\frac{\partial L}{\partial d_{22}}b_{12}
$$
$$
\frac{\partial L}{\partial a_{22}}
=
\frac{\partial L}{\partial d_{21}}b_{21}
+
\frac{\partial L}{\partial d_{22}}b_{22}
$$
可以把这些式子重新写成矩阵乘法：$$
\begin{bmatrix}
\dfrac{\partial L}{\partial d_{11}}&
\dfrac{\partial L}{\partial d_{12}}\\
\dfrac{\partial L}{\partial d_{21}}&
\dfrac{\partial L}{\partial d_{22}}
\end{bmatrix}
\begin{bmatrix}
b_{11}&b_{21}\\
b_{12}&b_{22}
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\partial L}{\partial a_{11}}&
\dfrac{\partial L}{\partial a_{12}}\\
\dfrac{\partial L}{\partial a_{21}}&
\dfrac{\partial L}{\partial a_{22}}
\end{bmatrix}
$$
即：$$
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial d}\,b^T
$$
同样地，对 $b$ 有：$$
\frac{\partial L}{\partial b}
=
a^T\frac{\partial L}{\partial d}
$$
对 $c$ 有：$$
\frac{\partial L}{\partial c}
=
\operatorname{sum}_0\left(\frac{\partial L}{\partial d}\right)
$$
（这里指按第 0 维求和）
运行代码如下：
```python
dh = dlogits @ W2.T
cmp('h', dh, h)

dW2 = h.T @ dlogits
cmp('W2', dW2, W2)

db2 = dlogits.sum(0)
cmp('b2', db2, b2)
```

## 10. h-preact

`hpreact` 仅用于通过 `h = torch.tanh(hpreact)` 计算 `h`，`tanh` 的导数为：
$$
a=\tanh(z)=\frac{e^{2z}-1}{e^{2z}+1}
$$
$$
\frac{\partial a}{\partial z}
=
1-a^2
=
1-\tanh^2(z)
$$
结合这一结果与链式法则，可以得到 `loss` 对 `hpreact` 的偏导数：

```python
dhpreact = (1.0 - h ** 2) * dh
cmp('hpreact', dhpreact, hpreact)
```

## 11. bngain, bnraw, bnbias

`bngain`、`bnraw` 和 `bnbias` 仅用于通过

```python
hpreact = bngain * bnraw + bnbias
```

计算 `hpreact`。其中 `bngain` 和 `bnbias` 是批量归一化中的参数。
- `hpreact` 的形状是 `(32, 64)`
- `bngain` 的形状是 `(1, 64)`
- `bnraw` 的形状是 `(32, 64)`
- `bnbias` 的形状是 `(1, 64)`
这里必须正确处理 PyTorch 广播操作在反向传播中的影响：
```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True) # 求和用于消除广播带来的重复贡献
cmp('bngain', dbngain, bngain)

dbnraw = bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

dbnbias = dhpreact.sum(0, keepdim=True) # 求和用于消除广播带来的重复贡献
cmp('bnbias', dbnbias, bnbias)
```

## 12. bndiff, bnvar_inv: BatchNorm Layer

`bndiff` 和 `bnvar_inv` 仅用于通过

```python
bnraw = bndiff * bnvar_inv
```

计算 `bnraw`。它们都属于批量归一化过程。

- `bndiff` 的形状是 `(32, 64)`
- `bnvar_inv` 的形状是 `(1, 64)`
- `bnraw` 的形状是 `(32, 64)`

把广播的影响考虑进去后，这一步相对直接：

```python
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True) # 求和用于消除广播带来的重复贡献
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

# Step 1
dbndiff = bnvar_inv * dbnraw
cmp('bndiff', dbndiff, bndiff) # 当前不正确，后续步骤会补全
```

## 13. bnvar
`bnvar` 仅用于通过

```python
bnvar_inv = (bnvar + 1e-5) ** -0.5
```

计算 `bnvar_inv`。二者都属于批归一化过程。
这里涉及幂运算，因此需要使用幂函数的求导公式。

```python
dbnvar = (-0.5 * (bnvar + 1e-5) ** (-1.5)) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)
```

## 14. bndiff2

`bndiff2` 仅用于通过

```python
bnvar = 1 / (n-1) * (bndiff2).sum(0, keepdim=True)
```

计算 `bnvar`。二者属于批归一化过程。

- `bndiff2` 的形状是 $(32\times64)$
- `bnvar` 的形状是 $(1\times64)$

> [!keypoint]
> 前向传播中，对 `bndiff2` 某个维度求和，在反向传播中会变成沿同一维度复制或广播梯度。<br>
> 反过来，如果前向传播中进行了复制或广播，那么反向传播中就需要沿相应维度求和。

下面用一个小例子表示 `bndiff2` 与 `bnvar` 的关系：

$$
\begin{bmatrix}
a_{11}&a_{12}\\
a_{21}&a_{22}
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
b_1\\
b_2
\end{bmatrix}
=
\begin{bmatrix}
\frac1{n-1}(a_{11}+a_{21})\\
\frac1{n-1}(a_{12}+a_{22})
\end{bmatrix}
$$

现在已知 $b_1$ 和 $b_2$ 的梯度，也就是 `bnvar` 两列对应的梯度。我们需要把梯度反向传播到各个 $a$。$b_1$ 或 $b_2$ 的梯度要分别流向 $a$ 的第一列或第二列，并乘上缩放因子 $\dfrac1{n-1}$。结合链式法则，可以得到：

```python
dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)
```

## 15. bndiff

`bndiff` 在两个地方被使用：

- `bndiff2 = bndiff**2`
- `bnraw = bndiff * bnvar_inv`

因此计算 `dbndiff` 很方便。不过要记住，我们已经在第 $12$ 步计算了 `dbndiff` 的第一部分。现在只需要把两条路径的梯度相加：

```python
# Step 2
dbndiff += 2 * bndiff * dbndiff2
cmp('bndiff', dbndiff, bndiff)
```

## 16. bnmeani

`bnmeani` 仅用于通过

```python
bndiff = hprebn - bnmeani
```

计算 `bndiff`。

- `bnmeani` 的形状是 `(1, 64)`
- `bndiff` 的形状是 `(32, 64)`
- `hprebn` 的形状是 `(32, 64)`

这里发生了广播，也意味着 `bnmeani` 的同一个元素在多行中被重复使用。<br>
因此，可以把 `dbndiff` 沿样本行求和，再取负号，得到 `dbnmeani`。<br>
`hprebn` 与 `bndiff` 形状相同，所以可以先直接复制相应梯度。


```python
dbnmeani = (-1.0) * dbndiff.sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

# Step 1
dhprebn = dbndiff.clone() # 复制张量本身，而不是只复制引用
cmp('hprebn', dhprebn, hprebn) # 当前不正确，后续步骤会补全
```

## 17. hprebn

如第 $16$ 步所示，`hprebn` 用于计算 `bndiff`。<br>
它还通过下面的式子参与 `bnmeani` 的计算：

```python
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
```

- `hprebn` 的形状是 `(32, 64)`
- `bnmeani` 的形状是 `(1, 64)`

这与广播恰好相反。我们需要把 `bnmeani` 的梯度乘以 `1/n`，再复制到 `hprebn` 的所有行，并把这一部分加到已有的 `dhprebn` 上：

```python
dhprebn += (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
cmp('hprebn', dhprebn, hprebn)
```

## 18. embcat, W1, b1: 线性层

`embcat` 参与线性层：

```python
hprebn = embcat @ W1 + b1
```

这一过程与第 $9$ 步已经分析过的线性层非常相似，因此可以直接复用相同的计算方式：

```python
dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)

db1 = dhprebn.sum(0)
cmp('b1', db1, b1)
```

## 19. emb

在线性层 `hprebn = embcat @ W1 + b1` 中，`emb` 先经过：

```python
embcat = emb.view(emb.shape[0], -1)
```

得到 `embcat`。`embcat` 的形状是 `(32, 30)`，`emb` 的形状是 `(32, 3, 10)`，这只是一次 reshape 操作。因此可以直接复制梯度，并把它重新变形回原来的形状：

```python
# 把 32×30 重新变形为 32×3×10
demb = dembcat.view(n, -1, emb.shape[2])
cmp('emb', demb, emb)
```

## 20. C

在线性层 `hprebn = embcat @ W1 + b1` 之前，`C` 通过

```python
emb = C[Xb]
```

生成 `emb`。

- `emb` 的形状是 `(32, 3, 10)`
- `C` 的形状是 `(27, 10)`
- `Xb` 的形状是 `(32, 3)`

`Xb` 的每一行包含三个整数，每个整数指定要读取 `C` 的哪一行。
反向传播时，需要找出 `Xb` 中每个整数使用了 `C` 的哪一行，再把相应梯度传回该行。如果同一行被使用多次，到达该行的梯度还必须相加。

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]        # 字符对应的整数索引
        dC[ix] += demb[k, j] # 把该字符嵌入的梯度累加到 C 对应行

cmp('C', dC, C)
```

# Excersise2:  Simplifying the Loss

练习 1 对每个细节进行了深入分析，但这种逐节点实现远远不符合真实工程中的写法。前面反向传播所经过的步骤拆得过于细碎。例如，如果从更高层次观察损失函数，反向传播公式实际上可以大幅简化。可以把前面的“整体展开版”损失函数与下面的简化计算进行比较：

```python
# 前向传播

# 原来的写法：
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# 现在的写法：
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())
```

两种方法几乎没有数值差异，那么反向传播是否也能简化？理想情况下，`dlogits` 应当成为一个只依赖 `logits` 和 `Yb` 的更短表达式。
目标是求：
$$
\frac{\partial\text{loss}}{\partial l_i}
=
\frac{\partial}{\partial l_i}
\left[
-\log\left(
\frac{e^{l_y}}{\sum_j e^{l_j}}
\right)
\right]=
-\frac{\sum_j e^{l_j}}{e^{l_y}}
\cdot
\frac{\partial}{\partial l_i}
\left[
\frac{e^{l_y}}{\sum_j e^{l_j}}
\right]
$$
这边 $l_i$ 指的是 $logits \in \mathbb{R}^{32\times27}$ 的第 $i$ 个分量。$y$ 是正确答案。

取 softmax：$p_i=\dfrac{e^{l_i}}{\sum_j e^{l_j}}:=\dfrac{e^{l_{i}}}{S}$。
$$
\frac{\partial\text{loss}}{\partial l_i}
=
-\frac{1}{p_{y}}
\cdot
\frac{\partial p_{y}}{\partial l_i}
$$
## Case 1. $i=y$
$$ \frac{\partial p_y}{\partial l_y} = \frac{e^{l_y}S-e^{l_y}e^{l_y}} {S^2}  = \frac{e^{l_y}}S - \frac{e^{l_y}}S \frac{e^{l_y}}S $$
又$\dfrac{e^{l_y}}S=p_y$
所以：$$  \frac{\partial p_y}{\partial l_y} = p_y(1-p_y) $$代入上式得到此时
$$\frac{\partial loss}{\partial l_y} = p_y-1 $$
## Case 2. $i\neq y$
此时：

$$ \frac{\partial p_y}{\partial l_i} = -\frac{e^{l_y}e^{l_i}}{S^2}  = -\frac{e^{l_y}}S \frac{e^{l_i}}S = -p_yp_i $$
由链式法则：
$$ \frac{\partial loss}{\partial l_i} = -\frac1{p_y}(-p_yp_i)  = p_i $$
代码运行如下：
```python
# 反向传播

dlogits = F.softmax(logits, dim=1) # 沿 logits 的每一行计算 Softmax
dlogits[range(n), Yb] -= 1 # 在 dlogits 的正确类别位置减去 1
dlogits /= n               # 由于损失对批次取均值，因此梯度还要除以 n

cmp('logits', dlogits, logits) # 只能做到近似一致，最大误差约为 6e-9
```

# Excersise 3: Simplifying the BatchNorm

我们的目标是找到一个单独的运算，一次性完成整个 BatchNorm 方程组的反向传播。

我们已经有 `dhpreact`（论文中的 $y_i$ 对应位置的梯度），希望高效地计算 `dhprebn`。

```python
# 前向传播

# 原来的写法：
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# 现在的写法：
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('max diff:', (hpreact_fast - hpreact).abs().max())
```

先写出 BatchNorm 论文中的方程：
$$
\mu_B=\frac1m\sum_{i=1}^m x_i
$$
$$
\sigma_B^2
=
\frac1{m-1}
\sum_{i=1}^m(x_i-\mu_B)^2
\qquad
\small{(\text{使用贝塞尔校正})}
$$
$$
\hat{x}_i
=
\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$
$$
y_i
=
\gamma\hat{x}_i+\beta
\equiv
BN_{\gamma,\beta}(x_i)
$$
我们已经知道 $\dfrac{\partial l}{\partial y_i}$，希望据此得到$\dfrac{\partial l}{\partial x_i}$。
## step 1 : 
对于$y_i=\gamma\hat{x}_i+\beta$ , 可以得到：
$$
\frac{\partial l}{\partial\hat{x}_i}
=
\frac{\partial l}{\partial y_i}\cdot\gamma
$$
## step 2 : 

计算 $\dfrac{\partial l}{\partial\sigma^2}$ 时，要注意 $\hat{x}$ 中有很多个 $\hat{x}_i$。每一个 $\hat{x}_i$ 都依赖同一个 $\sigma^2$。因此需要把所有 $\hat{x}_i$ 对应的贡献求和：
$$
\frac{\partial l}{\partial\sigma^2}
=
\sum_i
\left(
\frac{\partial l}{\partial\hat{x}_i}
\cdot
\frac{\partial\hat{x}_i}{\partial\sigma^2}
\right)
=
\gamma
\sum_i
\frac{\partial}{\partial\sigma^2}
\left[
(x_i-\mu)(\sigma^2+\epsilon)^{-1/2}
\right]
\cdot
\frac{\partial l}{\partial y_i}=
\frac12\gamma
\sum_i
(x_i-\mu)(\sigma^2+\epsilon)^{-3/2}
\cdot
\frac{\partial l}{\partial y_i}
$$
## step 3 :

接下来计算 $\dfrac{\partial l}{\partial\mu}$。$\mu$ 与 $\hat{x}$ 的关系有 $32$ 条路径，这与前面 $\sigma^2$ 和 $\hat{x}$ 的关系类似。同时，$\mu$ 与 $\sigma^2$ 之间还有一条路径，因为 $\sigma^2$ 对于每个特征维度而言是一个汇总统计量。所有这些进入 $\mu$ 的梯度都需要相加：
$$
\frac{\partial l}{\partial\mu}
=
\sum_i
\left(
\frac{\partial l}{\partial\hat{x}}
\cdot
\frac{\partial\hat{x}_i}{\partial\mu}
\right)
+
\left(
\frac{\partial l}{\partial\sigma^2}
\cdot
\frac{\partial\sigma^2}{\partial\mu}
\right)
$$
## step 4 :

现在，对 $x$ 中的每个 $x_i$，都有三类路径从它出发：
- 一条指向 $\mu$
- 一条指向 $\sigma^2$
- 一条指向对应的 $\hat{x}_i$
```python
# 反向传播
dhprebn = bngain * bnvar_inv / n * (n * dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
cmp('hprebn', dhprebn, hprebn) # 只能做到近似一致，最大误差约为 9e-10
```

