# 技术时间线

本时间线选择改变了“视频如何表示、生成或被用于决策”的工作。年份以论文首次公开或官方发布为主，不代表同年所有工作具有相同成熟度。

## 1990s–2003：显式运动与统计动态

### 关键思想

- 运动补偿、光流、morphing 和 image-based rendering。
- 将视频看作可重组的帧或可辨识的动态系统。
- 显式模拟器通过几何、材质和物理状态生成画面。

### 代表节点

- **1981 — Lucas–Kanade [[1]](#ref-1)**：用局部最小二乘估计光流。
- **1981 — Horn–Schunck [[2]](#ref-2)**：用全局平滑正则化估计密集光流。
- **2000 — Video Textures [[3]](#ref-3)**：通过寻找相似帧和跳转点，将短视频重组为连续动态纹理。
- **2003 — Dynamic Textures [[4]](#ref-4)**：以线性动力系统建模烟、火、水面和树叶等随机视觉过程。

### 留下的遗产

现代模型中的 flow-guided generation、warping、显式相机控制、state-space model 和数字孪生，都能在这一时期找到祖先。

## 2014–2016：深度视频预测

### 关键思想

- 使用 CNN 编码空间内容，用 RNN/LSTM 建模时间。
- 将自监督下一帧预测作为学习视觉表示的方法。
- 从直接预测像素转向预测变换、运动核和多尺度残差。

### 代表节点

- **2014 — Video sequence modeling with RNNs**：探索用递归网络预测视频。
- **2015 — ConvLSTM [[5]](#ref-5)**：把 LSTM 的全连接运算替换为卷积，保留空间结构。
- **2015 — Beyond MSE [[6]](#ref-6)**：指出 MSE 会产生平均化和模糊，引入多尺度与对抗损失。
- **2016 — CDNA / DNA / STP [[7]](#ref-7)**：通过预测像素变换和运动核学习物理交互。
- **2016 — PredNet [[8]](#ref-8)**：以预测编码思想组织深层视频预测网络。

### 核心矛盾

未来通常不是唯一的。确定性网络在多个合理未来之间求平均，因此“数值误差较低”可能对应“视觉结果更模糊”。

## 2016–2019：VAE 与 GAN 视频生成

### 关键思想

- 使用随机 latent 表达不确定未来。
- 用时空判别器替代逐像素损失。
- 解耦静态内容和动态运动。

### 代表节点

- **2016 — Generating Videos with Scene Dynamics [[9]](#ref-9)**：使用时空卷积 GAN，并分离前景和背景。
- **2017 — MoCoGAN [[10]](#ref-10)**：把内容 latent 和运动 latent 分开建模。
- **2018 — Stochastic Video Generation [[11]](#ref-11)**：进一步研究可控、随机和长程的视频生成。
- **2019 — DVD-GAN [[12]](#ref-12)**：展示大型 GAN 在高分辨率视频上的扩展能力。

### 核心矛盾

GAN 能产生锐利画面，但大规模训练不稳定、模式覆盖难以评估，也缺少像 likelihood 那样统一的训练目标。

## 2017–2023：视觉 Token 与 Transformer

### 关键思想

- 先压缩视频，再对离散 token 建模。
- 使用自回归、masked token prediction 和时空注意力。
- 将文本、图像和视频放入更统一的序列建模框架。

### 代表节点

- **2017 — VQ-VAE [[13]](#ref-13)**：学习离散视觉 codebook。
- **2021 — VideoGPT [[14]](#ref-14)**：VQ-VAE 视频 tokenizer + GPT 式自回归 Transformer。
- **2022 — Phenaki [[15]](#ref-15)**：用 causal video tokenizer 和 masked Transformer 生成可变长度视频。
- **2022/2023 — MAGVIT [[16]](#ref-16)**：用 3D tokenizer 和 masked generation 统一多种视频生成任务。
- **2023 — MAGVIT-v2 [[17]](#ref-17)**：继续提升视觉 tokenizer 和语言模型兼容性。

### 核心矛盾

压缩减少计算，却可能丢失运动细节；自回归生成具有统一概率建模形式，但视频 token 数量和串行采样成本非常高。

## 2020–2024：Diffusion 成为主线

### 关键思想

- 从噪声逐步还原视频分布。
- 复用大规模图像生成模型，再增加 temporal layer 或时空注意力。
- 在 latent space 训练，并通过级联超分辨率提高输出质量。

### 代表节点

- **2020 — DDPM [[18]](#ref-18)**：奠定现代 diffusion 生成框架。
- **2022 — Video Diffusion Models [[19]](#ref-19)**：系统展示时空 diffusion 在视频生成和预测中的能力。
- **2022 — Make-A-Video [[20]](#ref-20) / Imagen Video [[21]](#ref-21)**：大规模文本视频生成与级联时空超分辨率。
- **2022/2023 — Latent Video Diffusion [[22]](#ref-22)**：将生成过程移动到压缩 latent。
- **2023 — AnimateDiff [[23]](#ref-23)**：向图像 diffusion 注入可复用 motion module。
- **2023 — Stable Video Diffusion [[24]](#ref-24)**：开放权重图像到视频模型推动社区复现。
- **2024 — Lumiere [[25]](#ref-25)**：以 Space-Time U-Net 直接生成完整视频时间范围。
- **2024 — Sora [[26]](#ref-26)**：把不同时长、尺寸和宽高比的视频表示为空时 patch，并以 Transformer diffusion 扩展规模。

### 核心矛盾

短视频画质快速提高，但对象永久性、因果、复杂交互、超长状态一致性和推理成本仍是主要问题。

## 2018–2023：决策型 World Model 的并行谱系

这一谱系并不是从文本视频生成自然“升级”而来，而是来自控制、强化学习和规划。

### 代表节点

- **2018 — World Models [[27]](#ref-27)**：用 VAE 表示观测、RNN 建模 dynamics，并在想象环境中训练 controller。
- **2019 — PlaNet [[28]](#ref-28)**：在 latent dynamics 中进行在线规划。
- **2020 — Dreamer [[29]](#ref-29)**：在学习到的 latent world model 中优化行为。
- **2020 — MuZero [[30]](#ref-30)**：不重建完整观测，学习足以预测价值、策略和奖励的模型。
- **2023 — DreamerV3 [[31]](#ref-31)**：展示一套 world model 方法跨多种任务工作的可能性。
- **2023 — GAIA-1 [[32]](#ref-32)**：探索面向自动驾驶的生成式 world model。

### 核心矛盾

面向决策的 latent state 不一定需要生成漂亮像素；面向视频的生成器也不一定包含对规划有用的、因果一致的状态。

## 2024：生成模型与交互世界汇合

- **V-JEPA [[33]](#ref-33)**：在 representation space 预测缺失的时空信息，而非重建全部像素。
- **Genie 1 [[34]](#ref-34)**：从无动作标签的互联网游戏视频中发现 latent action，并生成可控制环境。
- **GameNGen [[35]](#ref-35)**：使用神经生成模型模拟可玩的游戏环境。
- **Sora**：提出大规模视频生成可能通向通用物理世界模拟器，同时公开展示模型的物理失败。

这一年之后，“视频生成模型是否就是 world model”成为领域的核心争论之一。

## 2025：动作、物理与实时交互

- **Cosmos 1 [[36]](#ref-36)**：提供面向 Physical AI 的 world foundation model、tokenizer 和数据处理平台。
- **V-JEPA 2 [[37]](#ref-37)**：将视频自监督表示、物理预测和少量机器人数据后训练连接到 zero-shot planning。
- **Veo 3**：除原生音视频生成外，研究显示其生成预训练中出现了分割、深度、物理属性和 affordance 等零样本能力。
- **Genie 3 [[38]](#ref-38)**：从文本生成可实时导航、维持数分钟的交互环境。
- **Sora 2**：加强物理、可控性和同步音频，同时更加突出肖像、来源与安全问题。
- **GWM-1 [[39]](#ref-39)**：将可探索世界、实时 avatar 和机器人动作条件 rollout 放在一个 general world model 家族下。

## 2026：Omnimodal Physical AI

- **Cosmos 3 [[40]](#ref-40)**：尝试用统一模型覆盖语言、图像、视频、音频、动作、forward dynamics、inverse dynamics 和机器人策略。
- **V-JEPA 2.1 [[41]](#ref-41)**：通过 dense predictive loss、deep self-supervision 和图像—视频输入路径强化时空 dense feature。
- **LeWorldModel [[42]](#ref-42) / EB-JEPA [[43]](#ref-43)**：一条路线研究从像素端到端学习可规划 latent dynamics，另一条路线提供单卡可复现的 JEPA 教学与研究组件。
- **Seedance 2.0、Kling 3.0 等创作模型**：进一步强化原生音视频、多模态参考、多镜头叙事和生成—编辑一体化。
- 研究重心从“生成一段看起来真实的视频”继续转向“能否记忆、交互、预测动作结果，并为真实策略带来收益”。

## 如何阅读这条时间线

建议不要只记模型名，而是为每个时代回答四个问题：

1. 视频被压缩或表示成了什么？
2. 时间变化是通过什么机制建模的？
3. 模型接受何种控制信号？
4. 成功是通过画质、预测误差，还是闭环任务完成率证明的？

JEPA 从图像表征、视频预测到动作条件规划的独立演化作为参考阅读收录，见 [JEPA 参考阅读](jepa.md)。

## 参考文献

<a id="ref-1"></a>[1] Bruce D. Lucas, and Takeo Kanade. [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI), 1981.

<a id="ref-2"></a>[2] Berthold K.P. Horn, and Brian G. Schunck. [Determining optical flow](https://doi.org/10.1016/0004-3702(81)90024-2). Artificial Intelligence, 1981.

<a id="ref-3"></a>[3] Arno Schödl, Richard Szeliski, David H. Salesin, and Irfan Essa. [Video textures](https://doi.org/10.1145/344779.345012). Proceedings of the 27th annual conference on Computer graphics and interactive techniques - SIGGRAPH '00, 2000.

<a id="ref-4"></a>[4] Gianfranco Doretto, Alessandro Chiuso, Ying Nian Wu, and Stefano Soatto. [Dynamic Textures](https://doi.org/10.1023/A:1021669406132). International Journal of Computer Vision, 2003.

<a id="ref-5"></a>[5] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, and Wang-chun Woo. [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). arXiv preprint, 2015.

<a id="ref-6"></a>[6] Michael Mathieu, Camille Couprie, and Yann LeCun. [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). arXiv preprint, 2015.

<a id="ref-7"></a>[7] Chelsea Finn, Ian Goodfellow, and Sergey Levine. [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157). arXiv preprint, 2016.

<a id="ref-8"></a>[8] William Lotter, Gabriel Kreiman, and David Cox. [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104). arXiv preprint, 2016.

<a id="ref-9"></a>[9] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612). arXiv preprint, 2016.

<a id="ref-10"></a>[10] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). arXiv preprint, 2017.

<a id="ref-11"></a>[11] Remi Denton, and Rob Fergus. [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687). arXiv preprint, 2018.

<a id="ref-12"></a>[12] Aidan Clark, Jeff Donahue, and Karen Simonyan. [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571). arXiv preprint, 2019.

<a id="ref-13"></a>[13] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). arXiv preprint, 2017.

<a id="ref-14"></a>[14] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). arXiv preprint, 2021.

<a id="ref-15"></a>[15] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399). arXiv preprint, 2022.

<a id="ref-16"></a>[16] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). arXiv preprint, 2022.

<a id="ref-17"></a>[17] Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737). arXiv preprint, 2023.

<a id="ref-18"></a>[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). arXiv preprint, 2020.

<a id="ref-19"></a>[19] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. [Video Diffusion Models](https://arxiv.org/abs/2204.03458). arXiv preprint, 2022.

<a id="ref-20"></a>[20] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792). arXiv preprint, 2022.

<a id="ref-21"></a>[21] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, et al. [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). arXiv preprint, 2022.

<a id="ref-22"></a>[22] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, et al. [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818). arXiv preprint, 2023.

<a id="ref-23"></a>[23] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). arXiv preprint, 2023.

<a id="ref-24"></a>[24] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). arXiv preprint, 2023.

<a id="ref-25"></a>[25] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). arXiv preprint, 2024.

<a id="ref-26"></a>[26] OpenAI. [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). Technical report, 2024.

<a id="ref-27"></a>[27] David Ha, and Jürgen Schmidhuber. [World Models](https://arxiv.org/abs/1803.10122). Advances in Neural Information Processing Systems 31, 2018.

<a id="ref-28"></a>[28] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). arXiv preprint, 2018.

<a id="ref-29"></a>[29] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). arXiv preprint, 2019.

<a id="ref-30"></a>[30] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, et al. [Mastering Atari, Go, chess and shogi by planning with a learned model](https://doi.org/10.1038/s41586-020-03051-4). Nature, 2020.

<a id="ref-31"></a>[31] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). arXiv preprint, 2023.

<a id="ref-32"></a>[32] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, et al. [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080). arXiv preprint, 2023.

<a id="ref-33"></a>[33] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, et al. [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). arXiv preprint, 2024.

<a id="ref-34"></a>[34] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). arXiv preprint, 2024.

<a id="ref-35"></a>[35] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). arXiv preprint, 2024.

<a id="ref-36"></a>[36] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). arXiv preprint, 2025.

<a id="ref-37"></a>[37] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). arXiv preprint, 2025.

<a id="ref-38"></a>[38] Google DeepMind. [Genie 3: A New Frontier for World Models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Project report, 2025.

<a id="ref-39"></a>[39] Runway. [Introducing Runway GWM-1](https://runway.com/research/introducing-runway-gwm-1). Project report, 2025.

<a id="ref-40"></a>[40] NVIDIA. [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800). arXiv preprint, 2026.

<a id="ref-41"></a>[41] Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mahmoud Assran, Koustuv Sinha, Michael Rabbat, et al. [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482). arXiv preprint, 2026.

<a id="ref-42"></a>[42] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312). arXiv preprint, 2026.

<a id="ref-43"></a>[43] Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, et al. [A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2602.03604). arXiv preprint, 2026.
