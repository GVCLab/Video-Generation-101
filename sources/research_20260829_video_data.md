# Research audit: video-generation data, data engines and governance

This file records the search, source verification and claim boundaries behind
[`resources/datasets.md`](../resources/datasets.md).

## Scope

- **Review date:** 2026-08-29 (Asia/Shanghai).
- **Review type:** focused scoping review, not a PRISMA systematic review or
  meta-analysis.
- **Primary question:** what data units, data-engine stages, release surfaces and
  governance controls are required to train and evaluate modern video generators
  and action-conditioned world models?
- **Time window:** foundational work before 2020 when needed, then 2020-01-01
  through 2026-08-29.
- **Evidence policy:** official proceedings or paper first; arXiv for work without
  proceedings; official repositories, project pages and dataset cards for the
  current release object, version, storage and terms. A wrapper's license badge is
  not treated as evidence that it owns upstream media rights.

## Search strategy

Queries combined `video generation dataset`, `video-text dataset`, `long video
caption dataset`, `UHD video dataset`, `spatial video dataset`, `robot trajectory
dataset`, `action dataset`, `physics video benchmark`, `data engine`,
`recaptioning`, `video deduplication`, `dataset license`, `opt out` and close
variants. Backward and forward citation chaining was used around WebVid,
HD-VILA-100M, InternVid, Panda-70M, OpenVid-1M and LAION-BVD.

The raw requests and responses were not archived as a reproducible search
bundle. Exact hit counts are therefore deliberately not reported: broad
bibliographic totals, ranked web results and direct-site lookups are not
comparable screening denominators.

| Source | Discovery surface | Use in the review |
|---|---|---|
| arXiv | focused title, abstract and version-history lookup | seed papers, recent preprints and withdrawal histories |
| OpenAlex / Crossref / DBLP | broad bibliographic and citation-neighbour discovery | candidate discovery, DOI and venue cross-check; not systematic screening |
| CVF / ICLR / NeurIPS / OpenReview / ACL / AAAI / RSS | direct paper lookup | authoritative venue, title and final-paper verification |
| GitHub / Hugging Face / ModelScope / project pages | direct release lookup | current media, metadata, storage, version, gate and terms |

## Inclusion and exclusion criteria

Included works changed or exposed at least one of the following:

1. the dominant sample unit or scale: source video, shot, clip, hour, frame,
   caption pair, trajectory or action segment;
2. shot/episode segmentation, quality filtering, motion filtering, recaptioning,
   deduplication, data mixing or curriculum;
3. long-duration, UHD, audio, spatial, depth, pose, action, force, tactile or
   physical supervision;
4. an actual public release surface: media, URL index, metadata, annotations,
   code, weights or a reproducible manifest;
5. a material governance mechanism: upstream license tracking, attribution,
   opt-out, removal, versioning, tombstones or decontamination.

Excluded or demoted items:

- dataset names found only in secondary lists without a primary record;
- model-training claims presented as though the training media were released;
- row counts presented as unique videos without checking the unit;
- repository license badges presented as blanket media rights;
- withdrawn duplicate records after the primary paper was identified;
- a new corpus whose only contribution was another scale number without a new
  data unit, annotation, access or governance contribution.

## The six release surfaces

The review treats these as separate evidence fields:

| Surface | What its existence proves | What it does not prove |
|---|---|---|
| Paper | authors describe a corpus or data engine | current access, reproducibility or rights |
| Project/repository | implementation or documentation exists | dataset bytes are present |
| URL/ID index | a locator list is available | URLs still resolve or media may be redistributed |
| Metadata/annotations | captions, timestamps or scores are downloadable | the source video is obtainable or licensed |
| Media | files can currently be downloaded | commercial training, redistribution or likeness rights |
| Training disclosure | a model was reportedly trained on a corpus | the corpus is public or independently auditable |

## Primary-source ledger: open-domain video and video-text data

| Work | Primary paper | Current official release surface | Verified distinction |
|---|---|---|---|
| HowTo100M | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Miech_HowTo100M_Learning_a_Text-Video_Embedding_by_Watching_Hundred_Million_Narrated_ICCV_2019_paper.pdf) | IDs/ASR-oriented resources; media remains platform-dependent | 1.22M source videos and 136.6M narrated clips are understanding-pretraining units, not visually grounded generation captions |
| WebVid | [Frozen in Time, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Bain_Frozen_in_Time_A_Joint_Video_and_Image_Encoder_for_ICCV_2021_paper.html) | [official repository and withdrawal notice](https://github.com/m-bain/webvid) | the maintainer no longer distributes URLs or captions after a cease-and-desist request |
| HD-VILA-100M | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xue_Advancing_High-Resolution_Video-Language_Representation_With_Large-Scale_Video_Transcriptions_CVPR_2022_paper.html) | [URL/timestamp metadata and downloader](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) | 3.3M sources, 100M clips and 371.5K hours; media is not hosted and the R-UDA is research/non-commercial |
| InternVid | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b7bfab38ed694b43e8c20c14f6c0e900-Abstract-Conference.html) | [gated 10M-FLT metadata card](https://huggingface.co/datasets/OpenGVLab/InternVid) | the paper's 7M videos / 234M clips / 760K hours, the data-card text and current viewer rows are different surfaces |
| Panda-70M | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Panda-70M_Captioning_70M_Videos_with_Multiple_Cross-Modality_Teachers_CVPR_2024_paper.html) | [metadata, timestamps and downloader](https://github.com/snap-research/Panda-70M) | the paper rounds to about 3.8M sources / 70.8M samples; the current full manifest reports 3,779,763 sources / 70,723,513 samples / 167K hours, while the estimated 36TB media remains upstream |
| Vript | [NeurIPS 2024 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6903a5aaece71b76623245fc6e32f01b-Abstract-Datasets_and_Benchmarks_Track.html) | [videos, annotations and license](https://github.com/mutonix/Vript) | 12K videos, over 420K clips and dense scripted captions; academic-only and no-redistribution terms |
| MiraData | [NeurIPS 2024 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/57f6683e550eb067936c9e9f0bcb8e31-Abstract-Datasets_and_Benchmarks_Track.html) | [repository](https://github.com/mira-space/MiraData), [dataset card](https://huggingface.co/datasets/TencentARC/MiraData), [supplementary terms / rights disclaimer](https://openreview.net/attachment?id=2myGfVgfva&name=supplementary_material) | the supplement says no right to copy, modify, publish, distribute or commercialize is implied absent a separate agreement; its conflict with repository/card wording leaves rights unresolved |
| LVD-2M | [NeurIPS 2024 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1df493ec1c2530c038d94d7300b5b368-Abstract-Datasets_and_Benchmarks_Track.html) | [URL/metadata and downloader](https://github.com/SilentView/LVD-2M) | about 2M long-take clips; source media is not directly distributed and terms inherit HD-VILA |
| OpenVid-1M | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0396ca5a4c628936609aa819bfbca916-Abstract-Conference.html) | [current media card](https://huggingface.co/datasets/nkp37/OpenVid-1M) | the card currently exposes 1,453,466 rows/12.4TB and 433K 1080p OpenVidHD items, but its CC-BY tag coexists with non-commercial and upstream-license restrictions |
| FineVideo | n/a, official release | [media card, provenance and removal terms](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | 43,751 CC-BY source videos, about 3,425 hours/600GB, with per-item provenance, opt-out and version updates |
| Koala-36M | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Koala-36M_A_Large-scale_Video_Dataset_Improving_Consistency_between_Fine-grained_Conditions_CVPR_2025_paper.html) | [current v1 metadata card](https://huggingface.co/datasets/Koala-36M/Koala-36M-v1), [repository and license](https://github.com/KlingAIResearch/Koala-36M) | the paper reports 36M clips / 172K hours; the current viewer exposes 3,766,054 rows, file shards estimate 35,961,606 rows / 48.9GB metadata, and media remains upstream under non-commercial terms |
| VideoUFO | [NeurIPS 2025 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1e6057620ed314b0020b3a30284b0f83-Abstract-Datasets_and_Benchmarks_Track.html) | [downloadable media/card](https://huggingface.co/datasets/WenhaoWang/VideoUFO) | 1.09M clips and 1,291 user-topic clusters; paper/card prose says about 800GB compressed media while the 2026-08-29 HF storage snapshot is 911GB; 0.29% is YouTube-ID overlap, not perceptual deduplication |
| UltraVideo | [NeurIPS 2025 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eeb3df2d70affd52f65ff3b9abb32487-Abstract-Datasets_and_Benchmarks_Track.html) | [official repository](https://github.com/xzc-zju/UltraVideo) | 58,781 UHD clips with structured captions; its custom license adds non-commercial restrictions |
| SpatialVID | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_SpatialVID_A_Large-Scale_Video_Dataset_with_Spatial_Annotations_CVPR_2026_paper.html) | [gated 7.67TB media/annotation card](https://huggingface.co/datasets/SpatialVID/SpatialVID) | 21K raw hours to 2.7M clips / 7,089 hours, with estimated camera poses, depth, dynamic masks and motion instructions; CC-BY-NC-SA |
| ViMix-14M | [arXiv:2511.18382](https://arxiv.org/abs/2511.18382) | [official card](https://huggingface.co/datasets/TimingYang/ViMix-14M), [file tree](https://huggingface.co/datasets/TimingYang/ViMix-14M/tree/main) | about 13.7M pairs / 22.8K hours; the current release is a 23.1GB full JSON manifest plus a 100-row example and source-specific download commands, not co-located integrated media |
| SceneScribe-1M | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_SceneScribe-1M_A_Large-Scale_Video_Dataset_with_Comprehensive_Geometric_and_Semantic_CVPR_2026_paper.html) | [author-named HF repository](https://huggingface.co/datasets/wangyunnan/SceneScribe-1M) | the paper reports 1M videos / 4,191 hours with caption, camera, depth, 3D tracks and motion masks; the current repository contains only a 2.54KB README, so its Apache badge is not corpus-license evidence |
| LAION-BVD | [arXiv:2608.24845](https://arxiv.org/abs/2608.24845) | [official downloads](https://projects.laion.ai/bvd/download.html), [current BVD-V-55M card](https://huggingface.co/datasets/laion/BVD-V-55M), [direct terms](https://github.com/LAION-AI/BVD/blob/main/assets/bvd_terms_of_use.pdf) | 1.3B public URL rows, 80M downloaded raw videos / 10M hours, and BVD-V's 55M clips from 2.4M sources / 41.1TB are distinct objects; gated media is research/non-commercial |

## Private or partially disclosed data engines

These are important technical evidence but are not public dataset releases.

| System | Primary source | Safe claim | Unsafe claim |
|---|---|---|---|
| Stable Video Diffusion | [official paper PDF](https://stability.ai/s/stable_video_diffusion.pdf) | authors describe about 580M annotated clips (577M clips in the table), a filtered 152M LVD-F training-example pool and filter/caption ablations | calling either count “source videos,” or claiming “LVD is downloadable” |
| Sora | [technical report](https://openai.com/index/video-generation-models-as-world-simulators/), [system card](https://openai.com/index/sora-system-card/) | the report supports recaptioning and native duration/resolution/aspect ratio; the card separately supports a mix of public, proprietary-partner and custom internal data | inferring named datasets, exact scale, mixture weights or item-level commercial rights |
| CogVideoX | [arXiv:2408.06072](https://arxiv.org/abs/2408.06072) | authors report about 35M filtered single-shot clips, average about six seconds, plus a negative-tag filter and dense recaptioning pipeline | “the 35M training corpus was released” |

## Action, robot and physical-data ledger

| Work | Primary source | Unit and supervision | Current release and terms | Boundary |
|---|---|---|---|---|
| Open X-Embodiment | [paper](https://arxiv.org/abs/2310.08864), [official repository](https://github.com/google-deepmind/open_x_embodiment) | over 1M trajectories across 22 embodiments | TFDS builders / GCS / RLDS; repository states Apache-2.0 for code and CC-BY-4.0 for other materials, while contributed datasets retain component terms | heterogeneous action spaces require normalization; video alone is not the action |
| DROID | [paper](https://arxiv.org/abs/2403.12945), [project/downloads](https://droid-dataset.github.io/droid/), [repository](https://github.com/droid-dataset/droid) | 76K trajectories, 350 hours and 564 scenes; current project reports 86 tasks | full RLDS about 1.7TB, raw about 8.7TB and a 2GB / 100-episode sample; repository MIT is a code license, not demonstrated blanket data permission | paper-era 84 versus current 86 tasks must carry version/date |
| RH20T | [paper](https://arxiv.org/abs/2307.00595), [official data page](https://rh20t.github.io/), [API](https://github.com/rh20t/rh20t_api) | 110K contact-rich sequences with vision, force, audio and action | official downloads plus API; API code is MIT, while data-media permission must be checked separately | embodiment, calibration and synchronization matter more than clip count |
| RoboMIND | [RSS 2025](https://www.roboticsproceedings.org/rss21/p152.pdf), [current v1.2 card](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) | 107K trajectories, 479 tasks, four embodiments and failure data | gated 12.3TB card with Apache-2.0 badge; gate conditions and component rights still apply | failures must not be silently filtered from world-model evaluation |
| RoboMIND 2.0 | [paper](https://arxiv.org/abs/2512.24653), [project](https://log2r.github.io/RoboMIND2.0/), [ModelScope release](https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0) | 310K dual-arm trajectories, six embodiments, 739 tasks and >1,000 hours, plus 12K tactile / 20K mobile / 20K simulated subsets | project links a public ModelScope release; current ModelScope terms require separate verification and do not inherit RoboMIND v1's badge | 2025 preprint; report real, tactile, mobile and simulated partitions separately |
| AgiBot World | [paper](https://arxiv.org/abs/2503.06669), [Beta card](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | over 1M trajectories / 2,976.4 hours, 217 tasks, 100+ scenarios across five domains | gated Beta media / proprioception / action release under CC-BY-NC-SA-4.0; gate requests contact details | version the Beta snapshot; author-reported policy gains are not independent data-quality validation |
| Action100M | [CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/EgoVis/html/Chen_Action100M_A_Large-scale_Video_Action_Dataset_CVPRW_2026_paper.html), [repository](https://github.com/facebookresearch/Action100M), [preview](https://huggingface.co/datasets/facebook/action100m-preview) | full-paper scale is 1.2M instructional source videos / 14.6 years / order-100M hierarchical action segments | current release is a 120,000-row, 10% video-level preview with YouTube IDs, metadata and nested nodes, not co-located media; FAIR Noncommercial Research License | open-vocabulary observed actions are not calibrated robot controls; the paper is formal, the full corpus is not yet the current release |
| CLEVRER | [paper](https://arxiv.org/abs/1910.01442) | controlled collision videos with descriptive, predictive, explanatory and counterfactual questions | project-dependent synthetic data release; re-check current terms before acquisition | narrow synthetic world, but high-quality causal labels |
| Physion | [paper](https://arxiv.org/abs/2106.08261), [project](https://physion-benchmark.github.io/) | eight physical scenarios with controlled contact prediction | official project release; current access terms must accompany a snapshot | simulator fidelity and scenario diversity bound external validity |
| PHYRE | [paper](https://arxiv.org/abs/1908.05656), [repository](https://github.com/facebookresearch/phyre) | interactive 2D physics tasks and actions | code/data tooling in the official repository; component terms apply | puzzle solving is not photorealistic video generation |
| NewtonBench-60K | [arXiv:2512.00425](https://arxiv.org/abs/2512.00425) | 50K training and 10K test videos across five Newtonian primitives | preprint-linked release surface; verify current files and terms before reuse | synthetic primitives; do not confuse it with the later LLM law-discovery benchmark of the same name |

## Version and count discrepancies retained in the chapter

| Item | Conflict | Handling rule |
|---|---|---|
| source video vs clip vs row | one source can yield many clips and multiple caption rows | preserve the publisher's unit; never compare naked numbers |
| InternVid | paper, data-card prose and current viewer expose different counts/surfaces | cite the surface beside the number |
| Panda-70M | paper rounds to about 3.8M sources / 70.8M samples; current full manifest reports 3,779,763 / 70,723,513 / 167K hours | use current exact manifest count for access, rounded paper count for history; neither proves 36TB media is hosted |
| OpenVid-1M | name says 1M; current card shows 1,453,466 rows and 12.4TB | report card snapshot with date; do not rename it OpenVid-2M |
| MiraData | named subsets, viewer rows and license texts disagree | mark rights as unresolved; do not infer commercial permission |
| Koala-36M | paper 36M, current viewer 3,766,054, shard estimate 35,961,606 | identify paper, viewer and manifest-estimate surfaces separately; do not call metadata hosted media |
| VideoUFO | paper/card prose says about 800GB, current HF storage is 911GB; 0.29% is source-ID overlap | date the storage snapshot and do not interpret source-ID overlap as perceptual or semantic decontamination |
| SceneScribe-1M | paper reports 1M / 4,191 hours, current author card has no data files | retain it as a paper milestone with an empty current release surface; do not inherit the README badge as corpus permission |
| Action100M | full-paper scale is order-100M action segments; current preview is 120K video rows | distinguish action-segment statistics from video-level manifest rows and record the 10% preview status |
| LAION-BVD | 1.3B URLs, 80M downloads and 55M BVD-V clips / 41.1TB | present separate units and access surfaces; cite the direct research/non-commercial terms |

No primary OpenVid-author, project, Hugging Face or arXiv record for an
“OpenVid-2M” dataset was found by the review date. `VideoGen-of-Thought`
([arXiv:2412.02259](https://arxiv.org/abs/2412.02259)) is a training-free
multi-shot generation framework, not a dataset; the duplicate
[arXiv:2503.15138](https://arxiv.org/abs/2503.15138) is withdrawn.

## Claim-handling rules

1. “Publicly reachable” is not “public domain,” “licensed for training,” or
   “licensed for redistribution.”
2. A wrapper can license only rights it controls; code, metadata and upstream
   media require separate fields.
3. URL availability, media availability and usable decoded yield are reported
   separately.
4. Caption density is not caption correctness. Captioner, version, prompt,
   frame-sampling policy and hallucination audit must be recorded.
5. Deduplication precedes split assignment and runs across all source datasets;
   source video, creator, episode and event groups are split together.
6. Aesthetic, CLIP, VLM, optical-flow or DOVER scores are model outputs, not
   ground truth. Thresholds require human calibration and rejection audits.
7. Dataset effects require fixed-compute ablations and multiple seeds. An
   author-reported downstream gain is not an independent dataset-quality score.
8. New 2026 preprints, especially LAION-BVD released four days before this
   review, are labelled provisional.

## Figure generation and verification

The review includes a generated data-engine figure plus a deterministic Mermaid
specification. The PNG gives the exact ten-stage main path and a compact linear
removal chain; the Mermaid adds the long feedback connections without obscuring
the main flow. The `scientific-schematics` Nano Banana path could not run
because `OPENROUTER_API_KEY` was not configured, so the built-in image-generation
tool was used as the documented fallback.

- **Use case:** `scientific-educational`.
- **Workspace asset:** `assets/diagrams/video-data-engine-v2.png`.
- **Dimensions:** 1672 × 941.
- **SHA-256:** `4c3d9835c097a853efa3efadcd5a46a54bd8f2106ba9aab4f74a4d10abc18169`.
- **Required main-path labels:** Source + Rights Gate; Immutable Ingest; Shot /
  Episode Split; Quality + Motion Filters; Cross-source Dedup + Benchmark
  Decontamination; Hierarchical Captions + Verification; Grouping + Mixing;
  Train / Eval Firewall; Versioned Split Manifests; Training + Audit.
- **Required governance labels:** Rights + Safety; Removal / Correction;
  Tombstone + New Manifest; Derived Data / Checkpoint Ledger.
- **Canonical order:** source / rights gate → immutable ingest → shot / episode
  split → technical + motion → content + safety → cross-source dedup → benchmark
  decontamination → hierarchical caption + verification → mix + group → train /
  eval firewall → versioned split manifests → training + audit.
- **Iteration:** a first edit preserved the old manifest/firewall order; a second
  added all ten numbered stages but misrouted three long feedback arrows. The
  accepted image keeps the exact main order and the unambiguous horizontal
  removal → tombstone → ledger chain, while the Mermaid alone carries the long
  feedback edges to source rights, manifests and training audit.
- **Visual inspection:** final project copy has no malformed text, cropped boxes,
  overlaps, logos or watermarks; the main flow, cross-cutting rights/safety plane,
  linear removal chain and train/eval boundary are visually distinct.

## Limitations

- This is a focused scoping review; raw search requests/responses were not archived
  as a reproducible bundle, so exact discovery hit counts and a PRISMA count are
  deliberately not reported.
- Dataset cards are mutable. Counts, storage, gating and terms are snapshots dated
  2026-08-29 and need re-checking before acquisition.
- Legal interpretation is outside this technical review. Conflicting or incomplete
  terms are marked as unresolved rather than resolved by inference.
- Link survival, decode yield, hash-level overlap and caption accuracy were not
  independently measured because the multi-terabyte media were not downloaded.
- Many industrial training corpora remain undisclosed, preventing independent
  reproduction of data-scale and data-mixture claims.
