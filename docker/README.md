# OS50 + AURA submission container

The exact inference container submitted to LISA 2026 Task 2. It runs both
branches on the ULF volume alone, fuses their probability maps at 0.6 / 0.4,
takes the argmax, and keeps the largest 26-connected component per label.

No high-field image, HF mask, or LF mask is used at inference — the only input
is the ULF MRI.

## What the container does

```
/input/**/*.nii.gz
   ├─ nnUNetv2_predict  -d 1 -tr nnUNetTrainerDiceTopK10_OS50_LS005  --save_probabilities
   └─ nnUNetv2_predict  -d 2 -tr nnUNetTrainerAURA_v0                --save_probabilities
        ↓
   0.6 · p(OS50) + 0.4 · p(AURA)  →  argmax  →  per-label largest CC
        ↓
/output/<case>_seg_prediction.nii.gz
```

Inputs are discovered recursively. Accepted filenames end in `_ciso.nii.gz`,
`_0000.nii.gz`, or plain `.nii.gz`; the case ID is the stem with `_ciso` or
`_0000` stripped. Duplicate case IDs are rejected rather than silently
overwritten. Output geometry — spacing, origin, direction — is copied from the
corresponding input volume, and a shape mismatch raises rather than writes a
misaligned mask.

Both branches use `checkpoint_best.pth` at fold 0. Test-time mirroring follows
each checkpoint's own `inference_allowed_mirroring_axes`: `(0,1,2)` for OS50,
`(0,1)` for AURA, whose left–right axis is disabled. No volume calibration,
class-specific thresholding, or probability rescaling is applied.

## Add the model weights

The two checkpoints are ~236 MB each and are not stored in this repository.
Place them at:

```
docker/models/
├── Dataset001_LISA/
│   └── nnUNetTrainerDiceTopK10_OS50_LS005__nnUNetPlans__3d_fullres/
│       ├── dataset.json          # results/training_logs/os50/dataset.json
│       ├── plans.json            # results/training_logs/os50/plans.json
│       └── fold_0/checkpoint_best.pth
└── Dataset002_LISA_VCF/
    └── nnUNetTrainerAURA_v0__nnUNetPlans__3d_fullres/
        ├── dataset.json          # results/training_logs/aura/dataset.json
        ├── plans.json            # results/training_logs/aura/plans.json
        └── fold_0/checkpoint_best.pth
```

The `dataset.json` and `plans.json` files are already in this repository under
[`results/training_logs/`](../results/training_logs/). Verify the checkpoints
against the recorded digests before building:

```bash
cd docker && sha256sum -c SHA256SUMS
```

## Build

Requires Docker with the NVIDIA Container Toolkit.

```bash
./build.sh                                   # default tag from MANIFEST.json
./build.sh registry.example.org/lisa-aura:v1 # or pass a full tag
```

The image is built `--platform linux/amd64` on
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`. `nnUNet/` inside this build
context must be an nnU-Net v2 tree with the AURA extensions already overlaid —
produced by `python scripts/install_extensions.py --nnunet-dir docker/nnUNet`
against a clean checkout. It is not committed here; the extension sources live
in [`nnunet_extensions/`](../nnunet_extensions/).

## Offline test

Networking is disabled to match the challenge runtime:

```bash
./test.sh <image> /absolute/path/to/input [output_dir]
```

Before submitting, check the output geometry, the filenames, and that the label
range is 0–11 with one prediction per input case.

## Push

```bash
docker login docker.synapse.org --username USERNAME   # PAT with Modify permission
./push.sh <image>
```

## Files

| File | Purpose |
| :-- | :-- |
| `Dockerfile` | Image definition; sets `nnUNet_results=/opt/nnunet_results` |
| `run_model.py` | Entry point: staging, both predictions, fusion, LCC, writing |
| `requirements.txt` | Pinned runtime dependencies |
| `MANIFEST.json` | Machine-readable record of models, weights, fusion and post-processing |
| `SHA256SUMS` | Digests of every packaged checkpoint and metadata file |
| `build.sh` / `test.sh` / `push.sh` | Build, offline GPU test, registry push |

## Checkpoint provenance

The packaged AURA checkpoint is the surviving **epoch-458** best checkpoint,
whose Dataset002→Dataset001 fold-0 cross-evaluation DSC is **0.7961**. The
intermediate snapshot behind the paper's standalone AURA DSC of 0.7950 was
overwritten rather than archived. Trainer, fold, architecture, 0.6/0.4 fusion,
mirroring metadata and LCC procedure are identical between them. This
distinction is recorded in `MANIFEST.json` and should be carried forward.
