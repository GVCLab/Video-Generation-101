# Video editing milestones research audit — 2026-08-29

## Scope

This audit supports `docs/tasks/video-to-video.md`. The goal was to reconstruct a capability-based history of video editing and connect it to current T2I, I2V, T2V, Diffusion / Flow, DiT, 3D/4D, instruction editing, memory and streaming methods.

## Retrieval note

The repository `research-lookup` skill was selected first, but both required backends were unavailable because `PARALLEL_API_KEY` and `OPENROUTER_API_KEY` were not configured. Following the skill fallback guidance, the audit used existing repository research plus fresh primary-source verification through arXiv, CVF, ACM/DOI and official project pages.

## Milestone selection rule

A work was treated as a proposed milestone only if it changed at least one of:

1. the editable unit, such as pixels, layers, objects, motion, camera or edit history;
2. the control interface, such as audio, masks, structure, text, references or instructions;
3. the temporal-consistency mechanism;
4. the training paradigm, such as per-video optimization, zero-shot, training-free or foundation-scale training;
5. the system role, such as a point tool, unified editor or iterative creative system.

Higher resolution or a stronger selected demo alone was not considered sufficient.

## Primary sources verified

- Video Rewrite, SIGGRAPH 1997: <https://doi.org/10.1145/258734.258880>
- Space-Time Video Completion, CVPR 2004 / TPAMI 2007: <https://graphics.stanford.edu/courses/cs448a-06-winter/wexler-completion-cvpr04.pdf>
- Video-to-Video Synthesis, NeurIPS 2018: <https://arxiv.org/abs/1808.06601>
- Layered Neural Atlases for Consistent Video Editing, SIGGRAPH Asia 2021: <https://arxiv.org/abs/2109.11418>
- Text2LIVE, ECCV 2022: <https://text2live.github.io/>
- Tune-A-Video, ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.html>
- Dreamix, 2023: <https://arxiv.org/abs/2302.01329>
- FateZero, ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html>
- Pix2Video, ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/html/Ceylan_Pix2Video_Video_Editing_using_Image_Diffusion_ICCV_2023_paper.html>
- TokenFlow, ICLR 2024: <https://arxiv.org/abs/2307.10373>
- AnyV2V, 2024: <https://arxiv.org/abs/2403.14468>
- Movie Gen, 2024: <https://arxiv.org/abs/2410.13720>
- VACE, 2025: <https://arxiv.org/abs/2503.07598>
- FiVE-Bench, ICCV 2025: <https://openaccess.thecvf.com/content/ICCV2025/html/Li_FiVE-Bench_A_Fine-grained_Video_Editing_Benchmark_for_Evaluating_Emerging_Diffusion_ICCV_2025_paper.html>
- Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset, 2025: <https://arxiv.org/abs/2510.15742>
- Consistent Video Editing as Flow-Driven Image-to-Video Generation, 2025: <https://arxiv.org/abs/2506.07713>
- VE-Bench, 2024: <https://arxiv.org/abs/2408.11481>
- IVEBench, 2025: <https://arxiv.org/abs/2510.11647>
- Memory-V2V, 2026: <https://arxiv.org/abs/2601.16296>
- EgoEdit, CVPR 2026: <https://arxiv.org/abs/2512.06065>
- FFP-300K, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FFP-300K_Scaling_First-Frame_Propagation_for_Generalizable_Video_Editing_CVPR_2026_paper.html>

## Evidence boundaries retained

- “Foundation editor” can mean a shared backbone, a model family or a unified interface; it does not imply one checkpoint performs every edit equally well.
- Training-free and inversion-free describe the editing procedure, not zero computation or guaranteed source fidelity.
- A plausible motion edit is not evidence of action-conditioned physical dynamics.
- VLM-based evaluators require calibration against people and targeted temporal diagnostics.
- 2026 works are included as current emerging nodes; their authors' reported advantages are not treated as independent cross-system conclusions.
