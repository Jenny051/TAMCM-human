# Volumetric Bodyfit

`volumetric_bodyfit` is a volumetric human body fitting toolkit. It loads scan meshes or preprocessed voxel data, predicts displacement fields for body-template vertices, and can run body model fitting, Chamfer refinement, surface-offset refinement, correspondence transfer, and evaluation reports.

This repository contains the refactored source code, lightweight runtime profiles, and small resource files. Large checkpoints and body model artifacts are intentionally kept outside this tree and mounted through environment variables.

## Project Layout

```text
.
├── src/volumetric_bodyfit/
│   ├── config/          # Runtime paths and environment variable handling
│   ├── dataflow/        # Dataset and Lightning DataModule code
│   ├── fieldnets/       # Voxel encoders and point-query networks
│   ├── solver/          # Training system, geometry utilities, and fitting logic
│   ├── entrypoints/     # Batch, CAPE, 4D-DRESS, and interactive entrypoints
│   ├── reports/         # Error reports and correspondence evaluation
│   └── resources/       # Pair lists and body vertex groups
├── profiles/            # Inference and evaluation runtime profiles
├── model_profiles/      # Model configuration profiles, without large checkpoints
├── artifact_manifest.csv
├── install.sh
├── setup.cfg
└── surface_asset_export.py
```

## Artifact Layout

Large files are not stored in this refactored source tree. At runtime, the code expects `BODYFIT_ARTIFACT_ROOT` to point to a folder with this structure:

```text
<artifact-root>/
├── storage/             # Checkpoint folders
└── support_data/        # Body models and support files
```

`artifact_manifest.csv` lists the large files that were left outside the refactored tree. The artifact names in that manifest are anonymized and are only meant to preserve count and size information.

## Environment Variables

Required:

```powershell
$env:BODYFIT_PROJECT_HOME = "<path-to-this-project>"
$env:BODYFIT_ARTIFACT_ROOT = "<path-containing-storage-and-support_data>"
```

Common optional overrides:

```powershell
$env:BODYFIT_OUTPUT_DIR = "<output-folder>"
$env:BODYFIT_CHECKPOINT_DIR = "<checkpoint-folder>"
$env:BODYFIT_MODEL_PROFILE_DIR = "<model-profile-folder>"
$env:BODYFIT_PROCESSED_DATA_DIR = "<processed-training-data>"
$env:BODYFIT_DEMO_SCAN_DIR = "<demo-scan-folder>"
$env:BODYFIT_ROTATED_DEMO_SCAN_DIR = "<rotated-demo-scan-folder>"
```

Dataset-specific optional overrides:

```powershell
$env:BODYFIT_CAPE_EVAL_DIR = "<cape-eval-folder>"
$env:BODYFIT_CAPE_RAW_DIR = "<cape-raw-folder>"
$env:BODYFIT_CAPE_SMPL_DIR = "<cape-body-model-info>"
$env:BODYFIT_DRESS_EVAL_DIR = "<dress-eval-folder>"
$env:BODYFIT_DRESS_SMPL_DIR = "<dress-body-model-info>"
$env:BODYFIT_FAUST_SCAN_DIR = "<faust-scan-folder>"
$env:BODYFIT_FAUST_REGISTRATION_DIR = "<faust-registration-folder>"
```

## Installation

Python 3.8 and a CUDA-compatible PyTorch environment are recommended. See `install.sh` for pinned dependency versions.

```powershell
cd "<path-to-this-project>"
pip install -e .
```

Full training or inference also requires these external components:

- PyTorch3D
- `human_body_prior`
- voxel preprocessing extensions
- body model fitting utilities compatible with the project interfaces
- a working CUDA runtime

## Main Entrypoints

Batch fitting:

```powershell
python -m volumetric_bodyfit.entrypoints.batch_register
```

CAPE workflows:

```powershell
python -m volumetric_bodyfit.entrypoints.cape_reconstruction
python -m volumetric_bodyfit.entrypoints.cape_scan_reconstruction
```

4D-DRESS workflow:

```powershell
python -m volumetric_bodyfit.entrypoints.dress_reconstruction
```

Interactive demo:

```powershell
streamlit run src/volumetric_bodyfit/entrypoints/interactive_app.py
```

Evaluation reports:

```powershell
python -m volumetric_bodyfit.reports.faust_report
python -m volumetric_bodyfit.reports.cape_report
python -m volumetric_bodyfit.reports.dress_report
python -m volumetric_bodyfit.reports.pair_transfer
```

## Configuration

Runtime profiles live in `profiles/`. Model profiles live in `model_profiles/`.

Hydra targets point to the refactored package structure:

```text
volumetric_bodyfit.dataflow.lightning_bridge.ShapeFieldDataModule
volumetric_bodyfit.dataflow.surface_samples.SmplFieldDataset
volumetric_bodyfit.solver.system.FieldRegistrationModule
```

To switch models, update `core.checkpoint` in one of the `profiles/*.yaml` files. The matching model configuration should exist at:

```text
model_profiles/<checkpoint-name>/config.yaml
```

The matching checkpoint archive should exist at:

```text
<checkpoint-folder>/<checkpoint-name>/checkpoints/*.zip
```

## Data Preparation

Training data is expected under:

```text
<processed-data>/<version>/stage_III/<split>/ifnet_indi/
```

Typical split names are `train`, `val`, and `test`. Each sample should include both vertex `.pt` files and voxel `.pt` files.

Inference inputs are selected by `profiles/*.yaml` and the environment variables above. Before running a full batch, test with a small number of `.ply` or `.obj` meshes to confirm paths, scale handling, and output folders.

## Outputs

The default output folder is:

```text
<project-home>/output/
```

Override it with `BODYFIT_OUTPUT_DIR` when needed. Common outputs include:

- aligned input meshes
- body-template fitting results
- Chamfer-refined results
- surface-offset refined results
- error logs and visualization meshes

## Development Checks

Syntax check:

```powershell
python -m compileall -q .
```

The refactored tree should not contain old package names, old file names, personal names, email addresses, or personal absolute paths.
