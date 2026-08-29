# MiniMax H3 timeline research note

Research date: 2026-08-29

## Primary sources

1. [MiniMax H3: An Open Model Breaking the Boundaries Between Tasks and Modalities](https://www.minimax.io/blog/minimax-h3), MiniMax, 2026-07-31.
2. [Open General Intelligence: MiniMax H3 Is Now Open Source](https://www.minimax.io/news/minimax-h3-open-source), MiniMax, 2026-08-03.

## Verified timeline facts

- H3 was first officially released on 2026-07-31 as a general-purpose omni-modal generation model.
- It accepts context composed of text, images, video and audio, and jointly generates video with native stereo audio.
- The official output specification is 4–15 seconds at 24 FPS with 32 kHz stereo audio. The Base workflow generates 768p; the complete hosted workflow can regenerate the result at up to 2K.
- H3-Base organizes modality-specific encodings as a packed multimodal sequence. A 33B dense, single-stream H3-Omni-Transformer jointly predicts video and audio latents. H3-VisualVAE and H3-AudioVAE provide separate visual and audio latent spaces.
- The initial open release provides two CFG-distilled task-specific Base checkpoints: FL2VA for text/first-or-last-frame conditioning and Ref2VA for multimodal references.
- Ref2VA supports up to 9 images, 3 video clips and 3 audio clips, with no more than 12 files in total.

## Evidence boundary used in the timeline

The phrase “open weight” applies to the H3-Base checkpoints, not automatically to the complete production system. H3-Context-IR and H3-Regenerate-2K remained hosted components at the initial release, and the native sparse-attention implementation was announced for a later update. Therefore local 768p Base inference and the official full 2K workflow are not treated as equivalent.
