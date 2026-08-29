# 精选阅读列表

本页不是追求数量的 awesome list，而是按概念演化挑选的阅读路径。建议优先阅读原始论文，并结合项目页中的视频与失败案例。

全部条目的标准引用、代码仓库类型和带日期的 GitHub Star 快照见 [引用与代码索引](bibliography.md)，也可直接下载 [完整 BibTeX](../bibliography/references.bib)。

本页把 JEPA 放在参考阅读中，而不是作为视频生成主线章节。若需要理解 latent prediction、V-JEPA 2-AC 与生成式 world model 的差别，可读 [JEPA 参考阅读](jepa.md)。

## 最小阅读集：8 篇建立全局观

1. **Video Textures [[1]](#ref-1)** — Schödl et al., 2000

   [Paper](https://dl.acm.org/doi/10.1145/344779.345012) · 理解数据驱动视频合成在深度学习之前如何工作。

2. **Deep Multi-Scale Video Prediction Beyond Mean Square Error [[2]](#ref-2)** — Mathieu et al., 2015

   [Paper](https://arxiv.org/abs/1511.05440) · 理解为什么像素 MSE 会导致模糊。

3. **Unsupervised Learning for Physical Interaction [[3]](#ref-3) through Video Prediction** — Finn et al., 2016

   [Paper](https://arxiv.org/abs/1605.07157) · 理解动作条件、像素变换和机器人视频预测。

4. **VideoGPT [[4]](#ref-4): Video Generation using VQ-VAE and Transformers** — Yan et al., 2021

   [Paper](https://arxiv.org/abs/2104.10157) · 理解视频 tokenizer 和自回归 Transformer。

5. **Video Diffusion Models [[5]](#ref-5)** — Ho et al., 2022

   [Paper](https://arxiv.org/abs/2204.03458) · 理解 diffusion 如何扩展到时间维。

6. **Video Generation Models as World Simulators [[6]](#ref-6)** — OpenAI, 2024

   [Technical report](https://openai.com/index/video-generation-models-as-world-simulators/) · 理解 spacetime patch、规模化和“world simulator”主张。

7. **World Models [[7]](#ref-7)** — Ha & Schmidhuber, 2018

   [Paper](https://arxiv.org/abs/1803.10122) · 理解决策型 world model 的 latent imagination 路线。

8. **Genie [[8]](#ref-8): Generative Interactive Environments** — Bruce et al., 2024

   [Paper](https://arxiv.org/abs/2402.15391) · 理解无动作标签视频、latent action 与交互生成的结合。

## A. 传统方法与状态空间

- **Lucas–Kanade optical flow** — Lucas & Kanade, 1981. [Paper](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf) [[9]](#ref-9)
- **Horn–Schunck optical flow** — Horn & Schunck, 1981. [DOI](https://doi.org/10.1016/0004-3702%2881%2990024-2) [[10]](#ref-10)
- **Video Textures** — Schödl et al., 2000. [Paper](https://dl.acm.org/doi/10.1145/344779.345012)
- **Dynamic Textures [[11]](#ref-11)** — Doretto et al., 2003. [DOI](https://doi.org/10.1023/A:1021669406132)

阅读问题：这些方法显式保留了哪些结构？现代生成模型又牺牲了哪些可解释性？

## B. 深度视频预测

- **Unsupervised Learning of Video Representations using LSTMs** — Srivastava et al., 2015. [Paper](https://arxiv.org/abs/1502.04681)
- **Convolutional LSTM Network [[12]](#ref-12)** — Shi et al., 2015. [Paper](https://arxiv.org/abs/1506.04214)
- **Beyond MSE** — Mathieu et al., 2015. [Paper](https://arxiv.org/abs/1511.05440)
- **Unsupervised Learning for Physical Interaction through Video Prediction** — Finn et al., 2016. [Paper](https://arxiv.org/abs/1605.07157)
- **Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning [[13]](#ref-13)** — Lotter et al., 2016. [Paper](https://arxiv.org/abs/1605.08104)

阅读问题：模型是在预测像素、运动，还是隐藏状态？多种合理未来如何表达？

## C. 生成模型路线：VAE 与 GAN

- **Generating Videos with Scene Dynamics** — Vondrick et al., 2016. [Paper](https://arxiv.org/abs/1609.02612)
- **MoCoGAN: Decomposing Motion and Content for Video Generation [[14]](#ref-14)** — Tulyakov et al., 2017. [Paper](https://arxiv.org/abs/1707.04993)
- **Stochastic Video Generation with a Learned Prior [[15]](#ref-15)** — Denton & Fergus, 2018. [Paper](https://arxiv.org/abs/1802.07687)
- **Adversarial Video Generation on Complex Datasets [[16]](#ref-16) / DVD-GAN** — Clark et al., 2019. [Paper](https://arxiv.org/abs/1907.06571)

阅读问题：内容与运动是否真的被解耦？判别器在检查单帧还是检查时间结构？

## D. Tokenizer 与 Transformer

- **Neural Discrete Representation Learning [[17]](#ref-17) / VQ-VAE** — van den Oord et al., 2017. [Paper](https://arxiv.org/abs/1711.00937)
- **VideoGPT** — Yan et al., 2021. [Paper](https://arxiv.org/abs/2104.10157)
- **Phenaki [[18]](#ref-18)** — Villegas et al., 2022. [Paper](https://arxiv.org/abs/2210.02399)
- **MAGVIT [[19]](#ref-19)** — Yu et al., 2022/2023. [Paper](https://arxiv.org/abs/2212.05199)
- **Language Model Beats Diffusion — Tokenizer Is Key to Visual Generation / MAGVIT-v2** — Yu et al., 2023. [Paper](https://arxiv.org/abs/2310.05737) [[20]](#ref-20)

阅读问题：压缩率、重建质量、token 数量和生成难度之间如何权衡？

## E. 生成模型路线：Diffusion、Flow 与大规模视频生成

- **Denoising Diffusion Probabilistic Models [[21]](#ref-21)** — Ho et al., 2020. [Paper](https://arxiv.org/abs/2006.11239)
- **Video Diffusion Models** — Ho et al., 2022. [Paper](https://arxiv.org/abs/2204.03458)
- **Make-A-Video [[22]](#ref-22)** — Singer et al., 2022. [Paper](https://arxiv.org/abs/2209.14792)
- **Imagen Video [[23]](#ref-23)** — Ho et al., 2022. [Paper](https://arxiv.org/abs/2210.02303)
- **Align Your Latents** — Blattmann et al., 2023. [Paper](https://arxiv.org/abs/2304.08818) [[24]](#ref-24)
- **AnimateDiff [[25]](#ref-25)** — Guo et al., 2023. [Paper](https://arxiv.org/abs/2307.04725)
- **Stable Video Diffusion [[26]](#ref-26)** — Blattmann et al., 2023. [Paper](https://arxiv.org/abs/2311.15127)
- **Lumiere [[27]](#ref-27)** — Bar-Tal et al., 2024. [Paper](https://arxiv.org/abs/2401.12945)
- **Sora technical report** — OpenAI, 2024. [Project](https://openai.com/index/video-generation-models-as-world-simulators/)

阅读问题：模型是在整段视频上联合去噪，还是分块、分帧或分辨率级联？时间层如何复用图像预训练？

## F. 决策型 World Model

- **World Models** — Ha & Schmidhuber, 2018. [Paper](https://arxiv.org/abs/1803.10122)
- **Learning Latent Dynamics for Planning from Pixels [[28]](#ref-28) / PlaNet** — Hafner et al., 2018/2019. [Paper](https://arxiv.org/abs/1811.04551)
- **Dream to Control [[29]](#ref-29) / Dreamer** — Hafner et al., 2019/2020. [Paper](https://arxiv.org/abs/1912.01603)
- **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model / MuZero** — Schrittwieser et al., 2019/2020. [Paper](https://arxiv.org/abs/1911.08265)
- **Mastering Diverse Domains through World Models [[31]](#ref-31) / DreamerV3** — Hafner et al., 2023. [Paper](https://arxiv.org/abs/2301.04104)
- **GAIA-1: A Generative World Model for Autonomous Driving [[32]](#ref-32)** — Hu et al., 2023. [Paper](https://arxiv.org/abs/2309.17080)

阅读问题：模型是否需要重建像素？它学习的是环境本身，还是仅足以预测价值与策略的状态？

## G. 大模型路线：视频基础模型与交互世界

- **Genie** — Bruce et al., 2024. [Paper](https://arxiv.org/abs/2402.15391)
- **GameNGen** — Valevski et al., 2024. [Paper](https://arxiv.org/abs/2408.14837) [[33]](#ref-33)
- **Cosmos World Foundation Model Platform** — NVIDIA, 2025. [Paper](https://arxiv.org/abs/2501.03575) [[34]](#ref-34)
- **Genie 3 [[35]](#ref-35)** — Google DeepMind, 2025. [Project](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- **GWM-1** — Runway, 2025. [Project](https://runway.com/research/introducing-runway-gwm-1) [[36]](#ref-36)
- **Cosmos 3 [[37]](#ref-37)** — NVIDIA, 2026. [Project](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)

阅读问题：动作是什么？状态保存多久？模型是否实时？规划收益是在模型内部还是真实环境中测量？

## H. 参考阅读：JEPA 路线

- **A Path Towards Autonomous Machine Intelligence [[38]](#ref-38)** — LeCun, 2022. [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
- **I-JEPA** — Assran et al., 2023. [Paper](https://arxiv.org/abs/2301.08243) [[39]](#ref-39)
- **MC-JEPA [[40]](#ref-40)** — Bardes et al., 2023. [Paper](https://arxiv.org/abs/2307.12698)
- **V-JEPA** — Bardes et al., 2024. [Paper](https://arxiv.org/abs/2404.08471) [[41]](#ref-41)
- **V-JEPA 2 [[42]](#ref-42)** — Assran et al., 2025. [Paper](https://arxiv.org/abs/2506.09985)
- **V-JEPA 2.1 [[43]](#ref-43)** — Mur-Labadia et al., 2026. [Paper](https://arxiv.org/abs/2603.14482)
- **LeJEPA [[44]](#ref-44) / LeWorldModel [[45]](#ref-45) / EB-JEPA / TD-JEPA [[47]](#ref-47)** — 训练稳定性、动作条件 latent dynamics 与强化学习延伸。详见 [JEPA 参考阅读](jepa.md)。

阅读问题：JEPA 预测的是像素、token 还是 latent？哪些工作只是表征学习，哪些工作真的接入了动作、rollout 或 planning？

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

本页主要参考工作：Mastering Atari, Go, chess and shogi by planning with a learned model [[30]](#ref-30)、A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures [[46]](#ref-46)。

## 参考文献

<a id="ref-1"></a>[1] [Video textures](https://doi.org/10.1145/344779.345012). Arno Schödl, Richard Szeliski, David H. Salesin, Irfan Essa. SIGGRAPH. 2000.

<a id="ref-2"></a>[2] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). Michael Mathieu, Camille Couprie, Yann LeCun. ICLR. 2016.

<a id="ref-3"></a>[3] [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157). Chelsea Finn, Ian Goodfellow, Sergey Levine. NeurIPS. 2016.

<a id="ref-4"></a>[4] [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas. arXiv preprint. 2021.

<a id="ref-5"></a>[5] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. NeurIPS. 2022.

<a id="ref-6"></a>[6] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.

<a id="ref-7"></a>[7] [World Models](https://arxiv.org/abs/1803.10122). David Ha, Jürgen Schmidhuber. arXiv preprint. 2018.

<a id="ref-8"></a>[8] [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. ICML. 2024.

<a id="ref-9"></a>[9] [An Iterative Image Registration Technique with an Application to Stereo Vision](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). Bruce D. Lucas, Takeo Kanade. IJCAI. 1981.

<a id="ref-10"></a>[10] [Determining optical flow](https://doi.org/10.1016/0004-3702%2881%2990024-2). Berthold K. P. Horn, Brian G. Schunck. Artificial Intelligence. 1981.

<a id="ref-11"></a>[11] [Dynamic Textures](https://doi.org/10.1023/A:1021669406132). Gianfranco Doretto, Alessandro Chiuso, Ying Nian Wu, Stefano Soatto. International Journal of Computer Vision. 2003.

<a id="ref-12"></a>[12] [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo. NeurIPS. 2015.

<a id="ref-13"></a>[13] [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104). William Lotter, Gabriel Kreiman, David Cox. ICLR. 2017.

<a id="ref-14"></a>[14] [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz. CVPR. 2018.

<a id="ref-15"></a>[15] [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687). Remi Denton, Rob Fergus. ICML. 2018.

<a id="ref-16"></a>[16] [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571). Aidan Clark, Jeff Donahue, Karen Simonyan. arXiv preprint. 2019.

<a id="ref-17"></a>[17] [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. NeurIPS. 2017.

<a id="ref-18"></a>[18] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://arxiv.org/abs/2210.02399). Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. ICLR. 2023.

<a id="ref-19"></a>[19] [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. CVPR. 2023.

<a id="ref-20"></a>[20] [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737). Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. ICLR. 2024.

<a id="ref-21"></a>[21] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Jonathan Ho, Ajay Jain, Pieter Abbeel. NeurIPS. 2020.

<a id="ref-22"></a>[22] [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792). Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. ICLR. 2023.

<a id="ref-23"></a>[23] [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, et al. arXiv preprint. 2022.

<a id="ref-24"></a>[24] [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818). Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, et al. CVPR. 2023.

<a id="ref-25"></a>[25] [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. ICLR. 2024.

<a id="ref-26"></a>[26] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-27"></a>[27] [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. SIGGRAPH Asia. 2024.

<a id="ref-28"></a>[28] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. ICML. 2019.

<a id="ref-29"></a>[29] [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. ICLR. 2020.

<a id="ref-30"></a>[30] [Mastering Atari, Go, chess and shogi by planning with a learned model](https://doi.org/10.1038/s41586-020-03051-4). Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, et al. Nature. 2020.

<a id="ref-31"></a>[31] [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. arXiv preprint. 2023.

<a id="ref-32"></a>[32] [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080). Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, et al. arXiv preprint. 2023.

<a id="ref-33"></a>[33] [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter. ICLR. 2025.

<a id="ref-34"></a>[34] [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, et al. arXiv preprint. 2025.

<a id="ref-35"></a>[35] [Genie 3: A New Frontier for World Models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Google DeepMind. Project report. 2025.

<a id="ref-36"></a>[36] [Introducing Runway GWM-1](https://runway.com/research/introducing-runway-gwm-1). Runway. Project report. 2025.

<a id="ref-37"></a>[37] [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800). NVIDIA. arXiv preprint. 2026.

<a id="ref-38"></a>[38] [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf). Yann LeCun. OpenReview working paper (v0.9.2). 2022.

<a id="ref-39"></a>[39] [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243). Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, et al. CVPR. 2023.

<a id="ref-40"></a>[40] [MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698). Adrien Bardes, Jean Ponce, Yann LeCun. arXiv preprint. 2023.

<a id="ref-41"></a>[41] [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, et al. arXiv preprint. 2024.

<a id="ref-42"></a>[42] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. arXiv preprint. 2025.

<a id="ref-43"></a>[43] [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482). Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, et al. arXiv preprint. 2026.

<a id="ref-44"></a>[44] [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544). Randall Balestriero, Yann LeCun. arXiv preprint. 2025.

<a id="ref-45"></a>[45] [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312). Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero. arXiv preprint. 2026.

<a id="ref-46"></a>[46] [A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2602.03604). Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, et al. arXiv preprint. 2026.

<a id="ref-47"></a>[47] [TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2510.00739). Marco Bagatella, Matteo Pirotta, Ahmed Touati, Alessandro Lazaric, Andrea Tirinzoni. ICLR. 2026.
