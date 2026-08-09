# 无条件视频生成

## 任务定义

无条件视频生成学习视频数据本身的分布，通常只给随机噪声或类别标签，输出一段新视频。它是研究视频分布建模的最纯粹形式，也是后来文本、图像、编辑和交互条件生成的底层能力来源。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统合成 | Video Textures [[1]](#ref-1)、Dynamic Textures [[2]](#ref-2) | 重组相似帧或学习线性动态系统 | 适合循环纹理，不适合开放语义 |
| 早期深度生成 | Video GAN [[3]](#ref-3)、MoCoGAN [[4]](#ref-4) | 时空卷积、内容/运动 latent 解耦 | 分辨率低、训练不稳定 |
| 大型 GAN | DVD-GAN [[5]](#ref-5)、DIGAN | 多尺度判别器、隐式动态建模 | mode collapse、难扩展到开放世界 |
| Token / AR | VQ-VAE [[6]](#ref-6)、VideoGPT [[7]](#ref-7)、MAGVIT [[8]](#ref-8) | video tokenizer + Transformer | token 数量大，采样慢 |
| Diffusion / DiT | DDPM [[9]](#ref-9)、Video Diffusion Models [[10]](#ref-10)、latent video diffusion [[11]](#ref-11) | 从噪声迭代还原视频分布 | 推理成本高，长程一致性难 |
| Foundation model | Sora [[12]](#ref-12)、Cosmos [[13]](#ref-13) | 大规模视频数据、spacetime patch、多任务后训练 | 数据治理、评测和可解释性复杂 |

## 技术演化逻辑

早期无条件生成主要关心“视频是否像真实数据”。Video Textures 和 Dynamic Textures 代表了深度学习前的数据重组与状态空间路线 [[1]](#ref-1), [[2]](#ref-2)。GAN 时代通过判别器获得锐利样本，但覆盖度与稳定性较弱 [[3]](#ref-3), [[4]](#ref-4), [[5]](#ref-5)。Tokenizer 与 Transformer 把视频转为序列，使似然建模和大规模自回归成为可能 [[7]](#ref-7), [[8]](#ref-8)。Diffusion 则把训练稳定性、样本质量和多样性推到新的水平，成为现代视频基础模型的主要底座 [[10]](#ref-10), [[14]](#ref-14), [[15]](#ref-15)。

今天无条件生成通常不再作为产品入口出现，因为纯随机视频很难控制；但它仍然是理解生成模型本体能力的关键：模型能否学到运动先验、场景统计、对象持久性和物理规律。

## 最新趋势

- 从像素级 GAN 转向 latent diffusion、DiT 和 flow matching。
- 从短视频样本转向长视频、可变宽高比和多分辨率时空 patch。
- 从单一生成目标转向可复用 backbone：同一个模型通过条件接口支持 T2V、I2V、V2V、inpainting 和 world modeling。
- 评测从 FVD/IS 扩展到物体持久性、物理常识、长程一致性和人工偏好。

## 关键问题

1. 无条件模型学到的是表面视频统计，还是可迁移的动态结构？
2. 如何评估 rare event 与长尾运动，而不是只看平均视觉质量？
3. 视频 tokenizer 的压缩损失会不会限制后续任务上限？
4. diffusion / flow 能否在保持质量的同时大幅降低采样成本？

## 参考文献

<a id="ref-1"></a>[1] [Video textures](https://doi.org/10.1145/344779.345012). Arno Schödl, Richard Szeliski, David H. Salesin, Irfan Essa. Proceedings of SIGGRAPH '00. 2000.

<a id="ref-2"></a>[2] [Dynamic Textures](https://doi.org/10.1023/A:1021669406132). Gianfranco Doretto, Alessandro Chiuso, Ying Nian Wu, Stefano Soatto. International Journal of Computer Vision. 2003.

<a id="ref-3"></a>[3] [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612). Carl Vondrick, Hamed Pirsiavash, Antonio Torralba. arXiv preprint. 2016.

<a id="ref-4"></a>[4] [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz. arXiv preprint. 2017.

<a id="ref-5"></a>[5] [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571). Aidan Clark, Jeff Donahue, Karen Simonyan. arXiv preprint. 2019.

<a id="ref-6"></a>[6] [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. arXiv preprint. 2017.

<a id="ref-7"></a>[7] [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas. arXiv preprint. 2021.

<a id="ref-8"></a>[8] [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. arXiv preprint. 2022.

<a id="ref-9"></a>[9] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Jonathan Ho, Ajay Jain, Pieter Abbeel. arXiv preprint. 2020.

<a id="ref-10"></a>[10] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. arXiv preprint. 2022.

<a id="ref-11"></a>[11] [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818). Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, et al. arXiv preprint. 2023.

<a id="ref-12"></a>[12] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.

<a id="ref-13"></a>[13] [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. arXiv preprint. 2025.

<a id="ref-14"></a>[14] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-15"></a>[15] [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. arXiv preprint. 2024.
