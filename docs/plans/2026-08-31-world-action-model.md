# World Action Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a teaching-oriented, research-survey chapter that defines World Action Models, situates them among adjacent topics, and links readers to the existing course map.

**Architecture:** Create one self-contained Markdown chapter under `docs/` with a progressive narrative: intuitive example, formal task contract, system loop, research routes, evaluation ladder, and research agenda. Update README navigation and the repository tree to expose the chapter without duplicating the detailed material in related pages.

**Tech Stack:** Markdown, internal relative links, existing repository documentation conventions.

---

### Task 1: Define the chapter boundary

**Files:**
- Create: `docs/world-action-model.md`
- Reference: `docs/world-models.md`
- Reference: `docs/tasks/action-conditioned-prediction.md`
- Reference: `docs/tasks/interactive-world-generation.md`

**Step 1:** Start with an intuitive action-consequence example and state the minimum WAM contract: observations, actions, state transition, uncertainty, and feedback.

**Step 2:** Add a boundary table contrasting video generation, video prediction, world models, action-conditioned prediction, interactive world generation, and WAM.

**Step 3:** Add internal links to the related chapters rather than duplicating their content.

### Task 2: Add teaching and survey content

**Files:**
- Modify: `docs/world-action-model.md`

**Step 1:** Describe the closed-loop architecture: observe, represent state, align action, predict multiple rollouts, verify, plan or replan.

**Step 2:** Summarize five research routes: action-conditioned video prediction, latent world models, game and embodied simulators, video foundation model reasoning, and interactive generation environments.

**Step 3:** Add an evaluation ladder from visual plausibility to counterfactual accuracy and closed-loop task utility.

### Task 3: Expose the chapter

**Files:**
- Modify: `README.md`

**Step 1:** Add World Action Model to the reader-entry table and the reasoning/world-model content-navigation row.

**Step 2:** Add it to the research learning path and repository tree.

**Step 3:** Verify all new local links resolve and the README does not retain the removed technical-evolution section.

### Task 4: Validate and deliver

**Files:**
- Test: `README.md`
- Test: `docs/world-action-model.md`

**Step 1:** Search for every `world-action-model.md` reference and verify its target exists.

**Step 2:** Run `git diff --check` and inspect the Markdown hierarchy.

**Step 3:** Commit the plan, chapter, and README updates; merge into `main`; push `main` to `origin`.
