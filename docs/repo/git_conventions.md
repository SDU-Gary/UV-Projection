# Git Conventions

## Remotes
- `origin`: personal repository (push target)
- `upstream`: original FaithC repository (sync source)

## Branches
- `main`: stable and reproducible states
- `dev`: integration branch for daily development
- `feat/<topic>`: feature development branches
- `exp/<topic>`: experiment branches that may be rebased/squashed

## Commit Rules
- Use short imperative subjects (`<= 72` chars).
- Keep each commit focused on one concern.
- Do not commit local build outputs or experiment run artifacts.

## Large File Policy
- `experiments/runs/` and `viewer/opengl_previewer/build/` must stay untracked.
- Raw assets (`assets/*.zip`, `assets/*.fbx`, top-level `assets/*.glb`) are ignored by default.
- If binary assets must be versioned, use Git LFS and document why in PR notes.
