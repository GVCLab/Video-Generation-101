# Timeline resource audit: nodes 01–25

Audit date: 2026-08-29. Scope: Lucas–Kanade through Phenaki, in the exact order used by `docs/timeline.md`. Only author/institution pages, papers, official repositories, and official model/download surfaces are listed. Recorded qualitative samples on a project page are not counted as an interactive demo. `none` means no qualifying public resource was found, not that no internal implementation ever existed.

## 01 — Lucas–Kanade

- paper_or_report: https://publications.ri.cmu.edu/an-iterative-image-registration-technique-with-an-application-to-stereo-vision-ijcai/
- project: none
- code: none
- weights: none
- demo: none
- notes: Foundational 1981 image-registration/optical-flow algorithm, not a trained model. No contemporaneous author/CMU project page or official implementation was found; later teaching code and third-party implementations are intentionally excluded.

## 02 — Horn–Schunck

- paper_or_report: https://doi.org/10.1016/0004-3702(81)90024-2
- project: none
- code: none
- weights: none
- demo: none
- notes: Classic variational optical-flow algorithm, not a trained model. No qualifying author/institution implementation was found; later reference and educational implementations are not official releases by Horn and Schunck.

## 03 — Video Rewrite

- paper_or_report: https://doi.org/10.1145/258734.258880
- project: none (legacy author page is offline)
- code: none
- weights: none
- demo: none
- notes: This is the legacy author-hosted project URL cited by the official Georgia Tech Video Textures page, but it now returns 404 and should be treated as retired. No official source-code or model release was found.

## 04 — Video Textures

- paper_or_report: https://www.microsoft.com/en-us/research/publication/video-textures/
- project: https://sites.cc.gatech.edu/gvu/perception/projects/videotexture/index.html
- code: none
- weights: none
- demo: none
- notes: The official Georgia Tech project page preserves the paper links and recorded result highlights. No author/institution source release or model weights were found; the method reorders captured frames rather than using learned weights.

## 05 — Dynamic Textures

- paper_or_report: https://doi.org/10.1109/ICCV.2001.937658
- project: http://www.cs.ucla.edu/~doretto/projects/dynamic-textures.html
- code: none
- weights: none
- demo: none
- notes: The paper and author materials identify this UCLA URL as the original data/movie project page, but the legacy page is no longer available. No currently retrievable author implementation was confirmed; later MATLAB/Python versions are third-party.

## 06 — Video (Language) Modeling

- paper_or_report: https://arxiv.org/abs/1412.6604
- project: https://ai.meta.com/research/publications/video-language-modeling-a-baseline-for-generative-models-of-natural-videos/
- code: none
- weights: none
- demo: none
- notes: Meta AI hosts an official publication record. No author/Meta training or inference repository, pretrained model, or interactive demo was found.

## 07 — ConvLSTM

- paper_or_report: https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html
- project: none
- code: none
- weights: none
- demo: none
- notes: No exact official implementation or pretrained checkpoint for the 2015 ConvLSTM paper was found. The author-maintained HKO-7 repository belongs to the later 2017 TrajGRU/benchmark paper and is therefore not labeled as the original ConvLSTM code.

## 08 — Beyond MSE

- paper_or_report: https://arxiv.org/abs/1511.05440
- project: http://cs.nyu.edu/~mathieu/iclr2016.html
- code: https://github.com/coupriec/VideoPredictionICLR2016
- weights: http://perso.esiee.fr/~coupriec/MathieuICLR16TestCode.zip
- demo: none
- notes: Official author Torch/Lua implementation. The linked ZIP is identified by the repository as containing two trained models and a UCF101 test subset; both the result page and archive are legacy author-hosted resources and may be intermittently unavailable.

## 09 — Action-Conditional Video Prediction

- paper_or_report: https://papers.nips.cc/paper_files/paper/2015/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html
- project: https://junhyuk.com/publication/2015_action_conditional/
- code: https://github.com/junhyukoh/nips2015-action-conditional-video-prediction
- weights: none
- demo: none
- notes: Official author project page and implementation. The author page links a recorded qualitative video at https://www.youtube.com/watch?v=4e-PqfpS8_4, but no downloadable pretrained checkpoint or interactive demo was found.

## 10 — DNA / CDNA / STP

- paper_or_report: https://arxiv.org/abs/1605.07157
- project: https://sites.google.com/site/robotprediction
- code: https://github.com/tensorflow/models/tree/5eb294f84bd3f415b548980e69fee63db1f6f1df/research/video_prediction
- weights: none
- demo: none
- notes: The official project page links the historical TensorFlow Models implementation and qualitative predictions. The directory was later removed from the current repository, so an immutable official snapshot is used; it is legacy TensorFlow 1.x code and no official pretrained checkpoint was located.

## 11 — PredNet

- paper_or_report: https://openreview.net/forum?id=B1ewdt9xe
- project: https://coxlab.github.io/prednet/
- code: https://github.com/coxlab/prednet
- weights: https://www.dropbox.com/s/iutxm0anhxqca0z/model_data_keras2.zip?dl=0
- demo: none
- notes: Official Cox Lab project, code, and Keras-2 model archive. The project page contains result videos but no hosted interactive demo; the software stack is now legacy Keras/TensorFlow.

## 12 — Video Pixel Networks

- paper_or_report: https://proceedings.mlr.press/v70/kalchbrenner17a.html
- project: none
- code: none
- weights: none
- demo: none
- notes: No DeepMind/author project page, source release, pretrained checkpoint, or demo specific to Video Pixel Networks was found. Repositories implementing PixelCNN or independently reproducing the paper are not official.

## 13 — SV2P

- paper_or_report: https://openreview.net/forum?id=rk49Mg-CW
- project: https://sites.google.com/site/stochasticvideoprediction/main
- code: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/video/sv2p.py
- weights: none
- demo: none
- notes: Official project samples and TensorFlow Tensor2Tensor implementation. Tensor2Tensor was archived and made read-only in 2023; no official pretrained checkpoint catalog or interactive demo was found.

## 14 — VGAN / Generating Videos with Scene Dynamics

- paper_or_report: https://proceedings.neurips.cc/paper_files/paper/2016/hash/04025959b191f8f9de3f924f0940515f-Abstract.html
- project: https://www.cs.columbia.edu/~vondrick/tinyvideo/
- code: https://github.com/cvondrick/videogan
- weights: https://drive.google.com/file/d/0B-xMJ5CYz_F9QS1BTE5yWl9aUWs/view?usp=sharing
- demo: none
- notes: Official author project, Torch7 implementation, and 1 GB pretrained-model archive. The project page hosts recorded generations, not an interactive demo; the Google Drive link is a legacy download and may require account/access handling.

## 15 — MoCoGAN

- paper_or_report: https://openaccess.thecvf.com/content_cvpr_2018/html/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.html
- project: https://github.com/sergeytulyakov/mocogan
- code: https://github.com/sergeytulyakov/mocogan
- weights: none
- demo: none
- notes: Official author implementation with qualitative GIFs. No official pretrained checkpoint or hosted interactive demo was found; PyTorch and Chainer reimplementations linked elsewhere are third-party.

## 16 — Video Generation from Text

- paper_or_report: https://arxiv.org/abs/1710.00421
- project: https://www.nec-labs.com/blog/video-generation-from-text/
- code: none
- weights: none
- demo: none
- notes: NEC Labs hosts an official publication/project record for this Duke/NEC work. No qualifying author/institution code repository, checkpoint, or interactive demo was found; similarly named public repositories are reproductions.

## 17 — SVG-LP

- paper_or_report: https://proceedings.mlr.press/v80/denton18a.html
- project: https://sites.google.com/view/svglp/
- code: https://github.com/edenton/svg
- weights: https://github.com/edenton/svg/tree/master/pretrained_models
- demo: none
- notes: Official author project and PyTorch repository. The repository includes pretrained SVG-LP models for BAIR and stochastic Moving MNIST; the project samples are recorded outputs, not a hosted interactive demo.

## 18 — Fréchet Video Distance

- paper_or_report: https://research.google/pubs/towards-accurate-generative-models-of-video-a-new-metric-challenges/
- project: https://research.google/blog/audio-and-visual-quality-measurement-using-fr%C3%A9chet-distance/
- code: https://github.com/google-research/google-research/tree/master/frechet_video_distance
- weights: https://www.kaggle.com/models/deepmind/i3d-kinetics
- demo: none
- notes: Official Google Research reference implementation. The listed weights are the official I3D Kinetics-400 feature extractor used by the metric, not generative-model weights; no interactive FVD service was found.

## 19 — DVD-GAN

- paper_or_report: https://arxiv.org/abs/1907.06571
- project: none
- code: none
- weights: none
- demo: none
- notes: No qualifying DeepMind/author project page, implementation, pretrained checkpoint, or interactive demo was found. Public DVD-GAN repositories are third-party reproductions and are intentionally excluded.

## 20 — VideoFlow

- paper_or_report: https://research.google/pubs/videoflow-a-conditional-flow-based-model-for-stochastic-video-generation/
- project: none
- code: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/video/next_frame_glow.py
- weights: none
- demo: none
- notes: Google Research publication and official Tensor2Tensor implementation. No separate project page was found. Tensor2Tensor was archived and made read-only in 2023; no official pretrained checkpoint or hosted demo was found.

## 21 — VQ-VAE

- paper_or_report: https://arxiv.org/abs/1711.00937
- project: https://deepmind.google/blog/deepmind-papers-at-nips-2017/
- code: https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
- weights: none
- demo: https://github.com/google-deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
- notes: DeepMind's official Sonnet VQ-VAE module and example notebook implement the quantizer and a trainable example, but they are not released paper checkpoints. The notebook is an executable official example rather than a hosted model endpoint.

## 22 — VideoGPT

- paper_or_report: https://arxiv.org/abs/2104.10157
- project: https://wilsonyan.com/videogpt/index.html
- code: https://github.com/wilson1yan/VideoGPT
- weights: https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/download.py
- demo: https://colab.research.google.com/github/wilson1yan/VideoGPT/blob/master/notebooks/Using_VideoGPT.ipynb
- notes: Official author project and repository. The download manifest provides official Google Drive identifiers for four VQ-VAE and two VideoGPT checkpoints; the Colab notebook is the official runnable demo. Community Hugging Face ports are not used as authoritative weights or demos.

## 23 — NÜWA

- paper_or_report: https://arxiv.org/abs/2111.12417
- project: https://github.com/microsoft/NUWA
- code: none
- weights: none
- demo: none
- notes: The official Microsoft repository is an archived project/documentation repository containing README material and assets, not training or inference implementation, so it must not be labeled Code. Microsoft hosts an official overview with recorded examples at https://www.microsoft.com/en-us/research/articles/nuwa/; no weights or interactive demo were released.

## 24 — CogVideo

- paper_or_report: https://arxiv.org/abs/2205.15868
- project: https://github.com/zai-org/CogVideo/tree/CogVideo
- code: https://github.com/zai-org/CogVideo/tree/CogVideo
- weights: https://github.com/zai-org/CogVideo/tree/CogVideo#download
- demo: https://models.aminer.cn/cogvideo/
- notes: The historical `CogVideo` branch is the official 2022 model release; the repository default branch now represents later CogVideoX work. Its download section links the official stage-1 and stage-2 archives. The AMiner surface is the institution-hosted demo and may have regional/account availability constraints.

## 25 — Phenaki

- paper_or_report: https://arxiv.org/abs/2210.02399
- project: https://sites.research.google/gr/phenaki/
- code: none
- weights: none
- demo: none
- notes: The official Google Research project page provides recorded samples. The paper states that the underlying models were not released; no author/Google code, checkpoint, or interactive endpoint was found, and community implementations are not official.
