# 精选阅读列表

本页不是追求数量的 awesome list，而是按概念演化挑选的阅读路径。建议优先阅读原始论文，并结合项目页中的视频与失败案例。

## 最小阅读集：8 篇建立全局观

1. **Video Textures** — Schödl et al., 2000

   [Paper](https://dl.acm.org/doi/10.1145/344779.345012) · 理解数据驱动视频合成在深度学习之前如何工作。

2. **Deep Multi-Scale Video Prediction Beyond Mean Square Error** — Mathieu et al., 2015

   [Paper](https://arxiv.org/abs/1511.05440) · 理解为什么像素 MSE 会导致模糊。

3. **Unsupervised Learning for Physical Interaction through Video Prediction** — Finn et al., 2016

   [Paper](https://arxiv.org/abs/1605.07157) · 理解动作条件、像素变换和机器人视频预测。

4. **VideoGPT: Video Generation using VQ-VAE and Transformers** — Yan et al., 2021

   [Paper](https://arxiv.org/abs/2104.10157) · 理解视频 tokenizer 和自回归 Transformer。

5. **Video Diffusion Models** — Ho et al., 2022

   [Paper](https://arxiv.org/abs/2204.03458) · 理解 diffusion 如何扩展到时间维。

6. **Video Generation Models as World Simulators** — OpenAI, 2024

   [Technical report](https://openai.com/index/video-generation-models-as-world-simulators/) · 理解 spacetime patch、规模化和“world simulator”主张。

7. **World Models** — Ha & Schmidhuber, 2018

   [Paper](https://arxiv.org/abs/1803.10122) · 理解决策型 world model 的 latent imagination 路线。

8. **Genie: Generative Interactive Environments** — Bruce et al., 2024

   [Paper](https://arxiv.org/abs/2402.15391) · 理解无动作标签视频、latent action 与交互生成的结合。

## A. 传统方法与状态空间

- **Lucas–Kanade optical flow** — Lucas & Kanade, 1981. [Paper](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf)
- **Horn–Schunck optical flow** — Horn & Schunck, 1981. [DOI](https://doi.org/10.1016/0004-3702%2881%2990024-2)
- **Video Textures** — Schödl et al., 2000. [Paper](https://dl.acm.org/doi/10.1145/344779.345012)
- **Dynamic Textures** — Doretto et al., 2003. [DOI](https://doi.org/10.1023/A:1021669406132)

阅读问题：这些方法显式保留了哪些结构？现代生成模型又牺牲了哪些可解释性？

## B. 深度视频预测

- **Unsupervised Learning of Video Representations using LSTMs** — Srivastava et al., 2015. [Paper](https://arxiv.org/abs/1502.04681)
- **Convolutional LSTM Network** — Shi et al., 2015. [Paper](https://arxiv.org/abs/1506.04214)
- **Beyond MSE** — Mathieu et al., 2015. [Paper](https://arxiv.org/abs/1511.05440)
- **Unsupervised Learning for Physical Interaction through Video Prediction** — Finn et al., 2016. [Paper](https://arxiv.org/abs/1605.07157)
- **Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning** — Lotter et al., 2016. [Paper](https://arxiv.org/abs/1605.08104)

阅读问题：模型是在预测像素、运动，还是隐藏状态？多种合理未来如何表达？

## C. VAE 与 GAN

- **Generating Videos with Scene Dynamics** — Vondrick et al., 2016. [Paper](https://arxiv.org/abs/1609.02612)
- **MoCoGAN: Decomposing Motion and Content for Video Generation** — Tulyakov et al., 2017. [Paper](https://arxiv.org/abs/1707.04993)
- **Stochastic Video Generation with a Learned Prior** — Denton & Fergus, 2018. [Paper](https://arxiv.org/abs/1802.07687)
- **Adversarial Video Generation on Complex Datasets / DVD-GAN** — Clark et al., 2019. [Paper](https://arxiv.org/abs/1907.06571)

阅读问题：内容与运动是否真的被解耦？判别器在检查单帧还是检查时间结构？

## D. Tokenizer 与 Transformer

- **Neural Discrete Representation Learning / VQ-VAE** — van den Oord et al., 2017. [Paper](https://arxiv.org/abs/1711.00937)
- **VideoGPT** — Yan et al., 2021. [Paper](https://arxiv.org/abs/2104.10157)
- **Phenaki** — Villegas et al., 2022. [Paper](https://arxiv.org/abs/2210.02399)
- **MAGVIT** — Yu et al., 2022/2023. [Paper](https://arxiv.org/abs/2212.05199)
- **Language Model Beats Diffusion — Tokenizer Is Key to Visual Generation / MAGVIT-v2** — Yu et al., 2023. [Paper](https://arxiv.org/abs/2310.05737)

阅读问题：压缩率、重建质量、token 数量和生成难度之间如何权衡？

## E. Diffusion、Flow 与大规模视频生成

- **Denoising Diffusion Probabilistic Models** — Ho et al., 2020. [Paper](https://arxiv.org/abs/2006.11239)
- **Video Diffusion Models** — Ho et al., 2022. [Paper](https://arxiv.org/abs/2204.03458)
- **Make-A-Video** — Singer et al., 2022. [Paper](https://arxiv.org/abs/2209.14792)
- **Imagen Video** — Ho et al., 2022. [Paper](https://arxiv.org/abs/2210.02303)
- **Align Your Latents** — Blattmann et al., 2023. [Paper](https://arxiv.org/abs/2304.08818)
- **AnimateDiff** — Guo et al., 2023. [Paper](https://arxiv.org/abs/2307.04725)
- **Stable Video Diffusion** — Blattmann et al., 2023. [Paper](https://arxiv.org/abs/2311.15127)
- **Lumiere** — Bar-Tal et al., 2024. [Paper](https://arxiv.org/abs/2401.12945)
- **Sora technical report** — OpenAI, 2024. [Project](https://openai.com/index/video-generation-models-as-world-simulators/)

阅读问题：模型是在整段视频上联合去噪，还是分块、分帧或分辨率级联？时间层如何复用图像预训练？

## F. 决策型 World Model

- **World Models** — Ha & Schmidhuber, 2018. [Paper](https://arxiv.org/abs/1803.10122)
- **Learning Latent Dynamics for Planning from Pixels / PlaNet** — Hafner et al., 2018/2019. [Paper](https://arxiv.org/abs/1811.04551)
- **Dream to Control / Dreamer** — Hafner et al., 2019/2020. [Paper](https://arxiv.org/abs/1912.01603)
- **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model / MuZero** — Schrittwieser et al., 2019/2020. [Paper](https://arxiv.org/abs/1911.08265)
- **Mastering Diverse Domains through World Models / DreamerV3** — Hafner et al., 2023. [Paper](https://arxiv.org/abs/2301.04104)
- **GAIA-1: A Generative World Model for Autonomous Driving** — Hu et al., 2023. [Paper](https://arxiv.org/abs/2309.17080)

阅读问题：模型是否需要重建像素？它学习的是环境本身，还是仅足以预测价值与策略的状态？

## G. 视频基础模型与交互世界

- **V-JEPA** — Bardes et al., 2024. [Paper](https://arxiv.org/abs/2404.08471)
- **Genie** — Bruce et al., 2024. [Paper](https://arxiv.org/abs/2402.15391)
- **GameNGen** — Valevski et al., 2024. [Paper](https://arxiv.org/abs/2408.14837)
- **Cosmos World Foundation Model Platform** — NVIDIA, 2025. [Paper](https://arxiv.org/abs/2501.03575)
- **V-JEPA 2** — Meta, 2025. [Project](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- **Genie 3** — Google DeepMind, 2025. [Project](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- **GWM-1** — Runway, 2025. [Project](https://runway.com/research/introducing-runway-gwm-1)
- **Cosmos 3** — NVIDIA, 2026. [Project](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)

阅读问题：动作是什么？状态保存多久？模型是否实时？规划收益是在模型内部还是真实环境中测量？

## 建议的论文笔记模板

```markdown
# Paper title

## Problem
这篇工作真正解决了什么？

## Representation
pixel / continuous latent / discrete token / structured state

## Temporal mechanism
recurrent / autoregressive / masked / diffusion / flow / state-space

## Conditions
text / image / video / audio / camera / action

## Evidence
数据、指标、人工评测、闭环任务分别证明了什么？

## Failure modes
作者展示了什么？还有什么没有测？

## Historical role
它改变了表示、架构、规模、控制还是评测？
```
