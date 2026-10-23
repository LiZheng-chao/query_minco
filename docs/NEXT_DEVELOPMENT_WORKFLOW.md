# Development workflow

Use this Git workspace as the sole future algorithm source. Keep historical research assets in place and read-only; never copy changed package files back into old experiments.

1. Create a feature branch for a real algorithm change, e.g. feature/query-decoder-v2, feature/physics-loss-v1, feature/teacher-search-v2.
2. Give semantic changes a new method ID and explicit component/config versions. A folder or branch rename alone does not change method identity.
3. Let new experiment wrappers import package modules. Store only configurations, immutable task/data bindings, entrypoints and necessary variant code in experiments/.
4. Write every run into an exclusive ignored research_outputs/<experiment>/<run>/ namespace. Bind commit_sha, method_id, config_sha, dataset_sha and checkpoint_sha plus the actual dependency source hashes.
5. Keep baseline contracts and checkpoints frozen. For historical compatibility, run bitwise regression before claiming equivalence.
6. Review explicit file paths before staging, inspect git diff --cached and file sizes/content, then commit locally. Push/PR is a separate human step.

Current limits: finish full R2/teacher-cache qualification and a separately audited controlled training driver before claiming full training pipeline portability. Keep the earlier teacher generation wrapper PROVENANCE_GAP visible. Choose the project license before publication.
