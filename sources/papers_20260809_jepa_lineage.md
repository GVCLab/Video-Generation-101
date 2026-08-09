# JEPA Lineage Research Audit

检索日期：2026-08-09

用途：支持 `docs/jepa.md` 的技术谱系、引用和官方代码核验。只记录论文、作者项目页和官方 GitHub 等一手来源；正文中的性能主张应回到原论文核对实验条件。

## Core architecture and mainline

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence** (2022): <https://openreview.net/forum?id=BZ5a1r-kVsf>
- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** (2023): <https://arxiv.org/abs/2301.08243>
- Official I-JEPA code: <https://github.com/facebookresearch/ijepa>
- Adrien Bardes, Jean Ponce, Yann LeCun, **MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features** (2023): <https://arxiv.org/abs/2307.12698>
- Adrien Bardes et al., **Revisiting Feature Prediction for Learning Visual Representations from Video** (2024): <https://arxiv.org/abs/2404.08471>
- Official V-JEPA code: <https://github.com/facebookresearch/jepa>
- Mahmoud Assran et al., **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** (2025): <https://arxiv.org/abs/2506.09985>
- Official V-JEPA 2 and V-JEPA 2-AC code: <https://github.com/facebookresearch/vjepa2>
- Lorenzo Mur-Labadia et al., **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning** (2026): <https://arxiv.org/abs/2603.14482>

## Stability, planning, and reinforcement-learning branches

- Randall Balestriero and Yann LeCun, **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics** (2025): <https://arxiv.org/abs/2511.08544>
- Official LeJEPA code: <https://github.com/galilai-group/lejepa>
- Lucas Maes et al., **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels** (2026): <https://arxiv.org/abs/2603.19312>
- Official LeWorldModel code: <https://github.com/lucas-maes/le-wm>
- Basile Terver et al., **A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures** (2026): <https://arxiv.org/abs/2602.03604>
- Official EB-JEPA code: <https://github.com/facebookresearch/eb_jepa>
- Marco Bagatella et al., **TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning** (2025): <https://arxiv.org/abs/2510.00739>
- Official TD-JEPA code: <https://github.com/facebookresearch/td_jepa>

## Verification notes

- I-JEPA's official repository is archived and read-only; it remains the authors' reference implementation.
- MC-JEPA has no paper-specific official GitHub repository identified as of the retrieval date.
- V-JEPA 2, V-JEPA 2-AC, and V-JEPA 2.1 share the `facebookresearch/vjepa2` repository.
- GitHub Star counts are not stored in this audit. The dated snapshot is generated from `bibliography/registry.json` into `bibliography/github-stars.json` and `docs/bibliography.md`.
- The configured academic research API was unavailable because no API key was present. Primary-source web lookup was used as the documented fallback.
