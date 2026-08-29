# Flow 与 Consistency：学习更直接的生成路径

Flow matching、rectified flow 和 consistency models 都试图更直接地连接简单噪声分布与数据分布，并减少生成所需的网络调用次数。它们关系密切但并非同义词：flow 学连续速度场，consistency 学同一生成轨迹上不同点应映射到一致结果。

## 1. Continuous Normalizing Flow

连续流用常微分方程描述样本演化：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t),\qquad t\in[0,1]
$$

$x_0$ 来自简单分布，$x_1$ 来自数据分布。若速度场正确，数值积分即可把噪声运输成视频。与离散 diffusion 的随机反向链相比，ODE 视角提供确定性轨迹和可直接使用的数值求解器。

## 2. Flow Matching [[1]](#ref-1)

直接学习边缘分布的真实速度场通常困难。Flow Matching 构造连接噪声样本 $x_0$ 与数据样本 $x_1$ 的条件概率路径，并回归该路径的条件速度：

$$
\mathcal{L}_{FM}=\mathbb{E}_{t,x_t}\|v_\theta(x_t,t)-u_t(x_t\mid x_1)\|^2
$$

训练不需要在每一步求解 ODE；求解主要发生在采样阶段。路径选择会影响速度场曲率和求解难度。

## 3. Rectified Flow [[2]](#ref-2)

Rectified flow 倾向使用更直的插值路径：

$$
x_t=(1-t)x_0+tx_1,\qquad \dot{x}_t=x_1-x_0
$$

网络学习沿路径的速度。轨迹越直，少步积分误差通常越小。Reflow 可用已有模型产生新的噪声—数据配对，再训练更直的流。但“直线插值”是训练构造，不意味着高维数据语义真的线性，也不保证一步生成毫无损失。

## 4. Consistency Models [[3]](#ref-3)

Consistency model 学习函数 $f_\theta(x_t,t)$，使同一概率流轨迹上的任意点都映射到同一个干净样本：

$$
f_\theta(x_t,t)\approx f_\theta(x_s,s),\quad x_t,x_s\text{ 位于同一轨迹}
$$

它可以从预训练 diffusion 蒸馏，也可独立训练，目标是一步或少步生成。多步 refinement 仍常用于提高质量，因此“consistency”不应自动等同于严格单步。

## 5. 与 diffusion 的关系

Diffusion、score-based model、probability flow ODE [[4]](#ref-4) 和 flow matching 在数学上有交叉：它们都学习如何让简单分布演化到数据分布。但训练目标、概率路径、参数化和采样方程不同。实践中还常使用 $\epsilon$、$x_0$、$v$ 等不同预测目标；看到“velocity prediction”并不能单独判定模型属于 rectified flow。

## 6. 为什么视频尤其需要这条路线

视频一次网络前向就很昂贵，多步扩散使延迟成倍增加。少步 flow/consistency 可以降低交互视频、预览、编辑和 world model rollout 的成本，也能减少长序列反复去噪造成的累计计算。真正目标不仅是帧率，还包括在少步下保持身份、运动速度、相机连续性和细节。

## 7. 视频实现要点

- 通常在时空 VAE latent 中学习速度场，先控制张量规模。
- DiT 用 spacetime patch 统一处理不同分辨率和长度；大规模图像侧的 rectified flow Transformer 实践见 [[5]](#ref-5)。
- 文本、首帧、动作等条件可通过 cross-attention 或条件 token 注入。
- 求解器可用 Euler、Heun 或更高阶方法；模型步数（NFE）应与真实延迟一并报告。
- 分段长视频仍需记忆、重叠窗口或关键帧锚定；少步采样本身不会解决长期状态问题。

## 8. 常见失败与评测

步数过少可能丢失高频细节、出现过平滑运动或条件偏移；蒸馏还可能继承教师偏差并降低多样性。评测应画出 NFE—质量—延迟曲线，而不是只报告最快设置；同时比较相同分辨率、时长、硬件和条件强度。

对视频应额外测量光流/轨迹连续性、身份与对象持久性、快速运动、镜头切换和长段落漂移。若用于交互 world model，还需报告动作到画面的响应延迟及闭环可控性。

## 9. 如何选择

若已有成熟 diffusion 教师且需要少步部署，可优先考虑 consistency 或其他蒸馏；若从头训练并希望使用 ODE 运输视角，flow matching/rectified flow 更自然；若最高质量比延迟重要，多步 diffusion 仍可能更稳。三者不是互斥标签，现代系统常把 flow 训练、蒸馏和少步求解器组合起来。

## 参考文献

<a id="ref-1"></a>[1] [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. ICLR. 2023.

<a id="ref-2"></a>[2] [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003). Xingchao Liu, Chengyue Gong, Qiang Liu. ICLR. 2023.

<a id="ref-3"></a>[3] [Consistency Models](https://arxiv.org/abs/2303.01469). Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever. ICML. 2023.

<a id="ref-4"></a>[4] [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. ICLR. 2021.

<a id="ref-5"></a>[5] [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, et al. ICML. 2024.
