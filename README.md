# InnoColor Edge — training

Orphan branch holding the training code for the InnoColor model family. This is
research code: unpolished, GPU-only, and not wired into the deployment app.
Nothing here is imported by `main` or `release/windows`.

For the system itself — installation, version lineage, usage — see
[`main`](https://github.com/Brocoliman/InnoColor/blob/main/README.md).

## Entry points

| Script | Trains |
|---|---|
| `train.py` | ResNet-18 generator, adversarially, against the Swin discriminator |
| `train_swin.py` | Swin Transformer generator — the teacher distilled for InnoColor Edge |
| `train_distiller.py` | The 2D LUT student, distilled from the teacher (`--dim`, default 17) |
| `test_distiller.py` | Evaluation for the distilled student |

Each script carries an example invocation in a comment on its first line.

## Supporting modules

| Path | What it is |
|---|---|
| `discrimintor_trs.py` | Swin Transformer: blocks, discriminator, and the `Generator_transformer_*` variants |
| `models_x.py` | 2D/3D LUT generators, classifiers, and TV regularizers |
| `datasets.py` | `ImageDataset`, `ImageDataset_single`, `ImageDataset_distiller` |
| `cvd_function.py` | CVD simulation matrices used to build training targets |
| `color_torch.py` | RGB/YUV conversion (BT.709) |
| `utils/` | 3D LUT generation, LUT evaluation, and distiller finetuning variants |

## Third-party code

`discrimintor_trs.py` is built on Microsoft's Swin Transformer implementation
(MIT licensed, © 2021 Microsoft, written by Ze Liu) and retains its original
header, with the InnoColor generator variants appended.
