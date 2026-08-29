# Timeline resource audit: nodes 26–50

- Audited: 2026-08-29
- Scope: indexes 26–50 in the current `docs/timeline.md`, from MAGVIT through MuZero.
- Source policy: paper/technical-report pages, author or institution project pages, author/institution GitHub repositories, and official model repositories only. Community reimplementations are deliberately excluded.
- Interpretation: `none` means that no first-party public resource of that type was identified; it does not prove that no internal resource exists.

## 26 — MAGVIT

- paper_or_report: https://arxiv.org/abs/2212.05199
- project: https://magvit.cs.cmu.edu/
- code: https://github.com/google-research/magvit
- weights: none
- demo: none
- notes: Google Research released the JAX source, but the repository is now archived/read-only. The maintainers explicitly state that the company did not approve release of the model weights: https://github.com/google-research/magvit/issues/16 . The project page is a results gallery, not a public inference service; the repository's “Colab” link does not currently identify a runnable notebook.

## 27 — MAGVIT-v2

- paper_or_report: https://arxiv.org/abs/2310.05737
- project: https://magvit.cs.cmu.edu/v2/
- code: none
- weights: none
- demo: none
- notes: No first-party MAGVIT-v2-specific implementation, weights, or hosted inference demo was identified. The Google Research `magvit` repository is official-related code for MAGVIT v1, not MAGVIT-v2 code, so it is not placed in the `code` field here. Open-MAGVIT2 and O2-MAGVIT2 are third-party reimplementations and must not be presented as the official release. Google Research publication record: https://research.google/pubs/language-model-beats-diffusion-tokenizer-is-key-to-visual-generation/

## 28 — VideoPoet

- paper_or_report: https://arxiv.org/abs/2312.14125
- project: https://sites.research.google/videopoet/
- code: none
- weights: none
- demo: none
- notes: The official project page is an interactive presentation of curated/pre-generated examples, not open model inference. Google has not published first-party code or weights for this model. Official Google Research article: https://research.google/blog/videopoet-a-large-language-model-for-zero-shot-video-generation/

## 29 — DDPM

- paper_or_report: https://arxiv.org/abs/2006.11239
- project: https://hojonathanho.github.io/diffusion/
- code: https://github.com/hojonathanho/diffusion
- weights: https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja
- demo: none
- notes: The author repository is the original TensorFlow 1.15 implementation and directly links the “Models and samples” archive above. No author-hosted live inference demo was identified.

## 30 — Latent Diffusion Models

- paper_or_report: https://arxiv.org/abs/2112.10752
- project: https://ommer-lab.com/research/latent-diffusion-models/
- code: https://github.com/CompVis/latent-diffusion
- weights: https://huggingface.co/CompVis/ldm-text2im-large-256
- demo: https://huggingface.co/spaces/CompVis/text2img-latent-diffusion
- notes: The linked checkpoint and Space are published by the official CompVis organization. The repository/model zoo contains additional task-specific LDM checkpoints; the link above is the representative text-to-image release.

## 31 — Video Diffusion Models

- paper_or_report: https://arxiv.org/abs/2204.03458
- project: https://video-diffusion.github.io/
- code: none
- weights: none
- demo: none
- notes: The first-party author project provides the paper and generated results but no public implementation, checkpoint, or inference service. Third-party PyTorch recreations, including `lucidrains/video-diffusion-pytorch`, are intentionally excluded.

## 32 — Make-A-Video

- paper_or_report: https://arxiv.org/abs/2209.14792
- project: https://makeavideo.studio/
- code: none
- weights: none
- demo: none
- notes: Meta's project was presented as work in progress; a form for possible future access did not constitute a public inference demo. No official code or weights were released. Official Meta announcement: https://ai.meta.com/blog/generative-ai-text-to-video/

## 33 — Imagen Video

- paper_or_report: https://arxiv.org/abs/2210.02303
- project: https://imagen.research.google/video/
- code: none
- weights: none
- demo: none
- notes: The official page explicitly says that the model and source code would not be released because of data/content and misuse concerns. The site is a results gallery, not a public inference demo.

## 34 — Rectified Flow / Flow Matching

- paper_or_report: Rectified Flow — https://arxiv.org/abs/2209.03003 ; Flow Matching — https://arxiv.org/abs/2210.02747
- project: https://ai.meta.com/research/publications/flow-matching-guide-and-code/
- code: https://github.com/gnobitab/RectifiedFlow
- weights: none
- demo: none
- notes: `gnobitab/RectifiedFlow` is the original authors' official Rectified Flow implementation. The Meta page is the later first-party guide/code project led by original Flow Matching author Yaron Lipman; its official PyTorch library is https://github.com/facebookresearch/flow_matching and explicitly states that it releases no pretrained models. Neither link is a video model checkpoint, because this node is the general generative-modeling foundation.

## 35 — Diffusion Transformer

- paper_or_report: https://arxiv.org/abs/2212.09748
- project: https://www.wpeebles.com/DiT
- code: https://github.com/facebookresearch/DiT
- weights: https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
- demo: https://huggingface.co/spaces/wpeebles/DiT
- notes: The author project links the official Meta repository, checkpoint, and author-owned Hugging Face Space. The GitHub repository is archived/read-only and uses a CC-BY-NC license. The official Space was in a runtime-error state when audited, but the URL is retained as the first-party demo endpoint.

## 36 — Latent Video Diffusion

- paper_or_report: https://arxiv.org/abs/2304.08818
- project: https://research.nvidia.com/labs/toronto-ai/VideoLDM/
- code: none
- weights: none
- demo: none
- notes: The original NVIDIA/LMU Video LDM project publishes the paper and result gallery only. No official implementation, checkpoint, or hosted inference demo was identified. Do not substitute later Stable Video Diffusion assets or community recreations for this node.

## 37 — AnimateDiff

- paper_or_report: https://arxiv.org/abs/2307.04725
- project: https://animatediff.github.io/
- code: https://github.com/guoyww/AnimateDiff
- weights: https://huggingface.co/guoyww/animatediff
- demo: none
- notes: The author repository identifies the project site and GitHub repository as the official sites and releases source plus pretrained motion modules. Colab notebooks linked from the README are community-hosted, so they are not labeled as an official demo here.

## 38 — Stable Video Diffusion

- paper_or_report: https://arxiv.org/abs/2311.15127
- project: https://stability.ai/news/stable-video-diffusion-open-ai-video-model
- code: https://github.com/Stability-AI/generative-models
- weights: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
- demo: none
- notes: Stability AI released first-party inference code and downloadable SVD/SVD-XT weights. Its hosted Stable Video Diffusion API was deprecated on 2025-07-24, so no current hosted demo is claimed. Official self-hosting/deprecation note: https://kb.stability.ai/knowledge-base/how-to-access-stable-video-diffusion

## 39 — W.A.L.T.

- paper_or_report: https://arxiv.org/abs/2312.06662
- project: https://walt-video-diffusion.github.io/
- code: none
- weights: none
- demo: none
- notes: The Google/Stanford author page is a result gallery; no official code, weights, or public inference service was identified. Google Research publication record: https://research.google/pubs/photorealistic-video-generation-with-diffusion-models/

## 40 — VideoCrafter1 / VideoCrafter2

- paper_or_report: VideoCrafter1 — https://arxiv.org/abs/2310.19512 ; VideoCrafter2 — https://arxiv.org/abs/2401.09047
- project: https://ailab-cvc.github.io/videocrafter2/
- code: https://github.com/AILab-CVC/VideoCrafter
- weights: https://huggingface.co/VideoCrafter/VideoCrafter2
- demo: https://huggingface.co/spaces/VideoCrafter/VideoCrafter
- notes: Tencent AI Lab CVC's official repository covers both releases and links their checkpoints. The VideoCrafter1 project is https://ailab-cvc.github.io/videocrafter1/ and the official Hugging Face organization lists all V1/V2 checkpoints at https://huggingface.co/VideoCrafter/models . The official Space was in a build-error state when audited; the repository also provides a local Gradio app.

## 41 — Lumiere

- paper_or_report: https://arxiv.org/abs/2401.12945
- project: https://lumiere-video.github.io/
- code: none
- weights: none
- demo: none
- notes: The author project is a gallery and paper page. No official Google implementation, weights, or inference endpoint was identified. Repositories such as `kyegomez/LUMIERE` and `lucidrains/lumiere-pytorch` are third-party recreations and should not be labeled official.

## 42 — Sora (2024 original)

- paper_or_report: https://openai.com/index/video-generation-models-as-world-simulators/
- project: https://openai.com/index/sora-is-here/
- code: none
- weights: none
- demo: none
- notes: Sora is a closed OpenAI model with no published source or downloadable weights. The original Sora product is unavailable as of 2026-04-26 according to the official launch page; Sora 2 is a distinct later model/product and should not be backfilled as the demo for this 2024 node.

## 43 — CogVideoX

- paper_or_report: https://arxiv.org/abs/2408.06072
- project: https://yzy-thu.github.io/CogVideoX-demo/
- code: https://github.com/zai-org/CogVideo
- weights: https://huggingface.co/zai-org/CogVideoX-5b
- demo: https://huggingface.co/spaces/zai-org/CogVideoX-5B-Space
- notes: The former THUDM URLs redirect to the official Z.ai organization. The repository, model card, and Space are first-party releases. CogVideoX-2B is Apache-2.0, while CogVideoX-5B uses the CogVideoX model license; avoid describing every checkpoint as identically licensed.

## 44 — Movie Gen

- paper_or_report: https://arxiv.org/abs/2410.13720
- project: https://ai.meta.com/research/movie-gen/
- code: none
- weights: none
- demo: none
- notes: Meta published the research report and first-party result galleries but no public code, checkpoints, or live inference service. Official Meta research record: https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/

## 45 — HunyuanVideo

- paper_or_report: https://arxiv.org/abs/2412.03603
- project: https://aivideo.hunyuan.tencent.com/
- code: https://github.com/Tencent-Hunyuan/HunyuanVideo
- weights: https://huggingface.co/tencent/HunyuanVideo
- demo: none
- notes: Tencent's official repository includes PyTorch model definitions, sampling code, pretrained weights, and a local Gradio server. The downloadable model is governed by the Tencent Hunyuan Community License rather than a standard OSI open-source license. No stable, clearly first-party hosted inference URL was identified for this audit.

## 46 — Wan 2.1

- paper_or_report: https://arxiv.org/abs/2503.20314
- project: https://wan.video/
- code: https://github.com/Wan-Video/Wan2.1
- weights: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
- demo: https://huggingface.co/spaces/Wan-AI/Wan2.1
- notes: These are the first-party Wan project, GitHub, Wan-AI model, and Wan-AI Space. The official repository released inference code and checkpoints and also links the organization's ModelScope collection.

## 47 — World Models

- paper_or_report: https://arxiv.org/abs/1803.10122
- project: https://worldmodels.github.io/
- code: https://github.com/hardmaru/WorldModelsExperiments
- weights: none
- demo: https://worldmodels.github.io/
- notes: The authors published the work as an interactive article with browser-based demonstrations and linked experiment code. The project URL therefore legitimately serves as both the explanatory project page and the interactive demo. No separate first-party pretrained-weight package was identified.

## 48 — PlaNet

- paper_or_report: https://arxiv.org/abs/1811.04551
- project: https://danijar.com/project/planet/
- code: https://github.com/google-research/planet
- weights: none
- demo: none
- notes: The official Google Research implementation is public but archived/read-only. No official pretrained weights or hosted interactive demo was identified.

## 49 — Dreamer

- paper_or_report: https://arxiv.org/abs/1912.01603
- project: https://danijar.com/project/dreamer/
- code: https://github.com/danijar/dreamer
- weights: none
- demo: none
- notes: This is the original implementation maintained by first author Danijar Hafner. No separate first-party pretrained-weight distribution or public inference demo was identified.

## 50 — MuZero

- paper_or_report: https://arxiv.org/abs/1911.08265
- project: https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
- code: https://gist.github.com/Mononofu/6c2d27ea1b3a9b3c1a293ebabed062ed
- weights: none
- demo: none
- notes: The code link is paper author Julian Schrittwieser's DeepMind-copyrighted MuZero pseudocode, not a complete runnable training implementation. DeepMind did not release full first-party source or pretrained weights. `muzero-general` is a community implementation and must not be labeled official.
