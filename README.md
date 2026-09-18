# InnoColor Edge: A Real-Time Wearable System with Distilled Vision Transformer Guided Recoloring for Color Vision Deficiency

An on-device system that recolors a live camera feed for viewers with color
vision deficiency (CVD), with a Tkinter interface for switching between model
versions, simulating CVD types, and building/editing datasets.

## Version lineage

The repository ships three generations of recoloring model. They are selectable
at runtime from the toolbar.

| Version | In-app name | Description | Publication |
|---|---|---|---|
| DaltNet | `Dalt-NET` | Earliest architecture; multi-mapping 3D LUT method | [IEEE ICMI, Wang, 2024](https://www.researchgate.net/publication/382209331_Creation_of_DALT-NET_Deep_Learning_3D_Lookup_Table_Generation_Approach_to_Daltonization_for_Dichromatic_Vision) |
| InnoColor (original) | `InnoColor` | Second architecture; salient-attention driven | [IEEE ICMI, Wang, 2025](https://www.researchgate.net/publication/395368307_Innocolor_a_Pioneering_Image_Recoloring_Framework_Driven_by_Salient_Attention_Networks) |
| InnoColor Edge | `Distillation` | **Current architecture;** Dense Distillation Perceptual Algorithm (DDPA): a ViT teacher distilled into a lookup-table student | Unpublished, see [paper](paper/innocolor-edge.pdf) |

A `Hue Rotation` baseline ([Huang et al., 2008](https://www.researchgate.net/publication/29622829_Enhancing_Color_Representation_for_the_Color_Vision_Impaired))
is also included for comparison.

## Platforms

| Branch | Platform | Status |
|---|---|---|
| `main` | Linux (Ubuntu) | Development. Canonical documentation lives here. |
| `release/windows` | Windows | Stable / production. Adds TCS3472 RGB ambient light sensor support. |
| `training` | — | Orphan branch; training code for the models above. |

Both deployments run the same models and the same interface. Differences are
limited to display scaling defaults, installation, and the ambient
sensor.

## Layout

```
innocolor/
│── main.py
│── settings.py           app and device (host OS and XBox controller) specifications
|── base/                 parent objects for utils
|── utils/                interface objects and app-specific model-running processes
|── interfaces/           the three Tk interfaces
|── icons/
|
|── dsX.pth               datasets: input/ground-truth pairs plus the model used
|
|── model/                daltnet/ innocolor/ distill/ hr/
|── checkpoint/           daltnet/ innocolor/ distill/
|── cvdsim/               CVD simulation
|
|── third_party/          vendored dependencies, unmodified
|   |── sam2/             Meta's SAM 2, used only as InnoColor's saliency backbone
|   |── sam2_configs/     Hydra configs for the above
|
|── rgb_sensor_display.py    (release/windows only)
|── rgb_sensor_print.ino.ino (release/windows only)
|
│── requirements.txt
│── README.md
```

`third_party/sam2` is a frozen, unmodified vendor drop of
[SAM 2](https://github.com/facebookresearch/sam2). It is reached only by
`model/innocolor/SAM2UNet.py`; no other model depends on it. Because the
vendored code imports itself by the absolute name `sam2`,
`model/innocolor/__init__.py` puts `third_party/` on the path. Nothing inside
`third_party/` has been edited, so it can still be diffed against upstream.

## Installation

**1. Clone**

```bash
git clone https://github.com/Brocoliman/InnoColor.git
cd InnoColor
```

For the Windows deployment, check out `release/windows` first.

**2. Create a conda environment**

Developed and tested with conda.

```bash
conda create -n innocolor python=3.11
conda activate innocolor
```

**3. Install PyTorch**

On Linux, PyTorch is installed by the requirements file in the next step and
this can be skipped. On Windows, install PyTorch and torchvision first,
following the instructions for your system at
[PyTorch Get Started](https://pytorch.org/get-started/locally/).

**4. Install the remaining dependencies**

```bash
pip install -r requirements.txt
```

**5. Download checkpoints**

From [Google Drive](https://drive.google.com/drive/u/0/folders/1mCRWx0PGKODix-e9XUzvoWWF8T7DReiZ),
into their respective folders under `checkpoint/`:

- `daltnet/` — `classifier_190.pth`, `LUTs_190.pth`
- `innocolor/` — `classifier_2000.pth`, `generator_200.pth`, `LUTs_2000.pth`, `SAM2UNet-SOD.pth`
- `distill/` — `generator_100.pth`

**6. Download the sample datasets**

`ds1.pth`, `ds2.pth`, `ds3.pth` into the repository root.

**7. Configure `settings.py`**

- Set `FOLDER` to the absolute path of this project.
- Set `WIDTH` and `HEIGHT` to the desired window size. The GUI was designed for
  a 3840 × 2160 display; the checked-in defaults differ per branch.
- `ASPECT` is the aspect ratio of captured/displayed images. Leaving it alone is
  recommended.
- `BOXHEIGHT` controls the image display size in the data-editing interface.
  Ensure `2 * ASPECT * BOXHEIGHT < WIDTH`, or displays will not fit.

## Run

```bash
python main.py
```

## Quick guide

The camera interface and the data-editing interface both carry the InnoColor
toolbar: `model`, `sim`, and `highlight`.

**`model`** is categorized between `[P]` protanopia-targeted, `[D]` deuteranopia-targeted:

- `None`: no daltonization
- `Hue Rotation`: `[P]`
- `Dalt-NET`: `[P]`
- `InnoColor`: `[P]`, note this is not the newest method
- `Distillation`: `[D]`, InnoColor Edge / DDPA, the current method

Models trained for protanopia are usually workable for deuteranopia and vice
versa, with significant limitations. In the data-editing interface each model
displays a parameter slider; the value is passed through but no current model
uses it.

**`sim`** is the simulated CVD, with a severity slider from 0 (no CVD) to 1 (full
dichromacy for affected cone).

- `None`
- `Protan`: red cone deficiency, second most common
- `Deutan`: green cone deficiency, most common
- `Tritan`: blue cone deficiency, much rarer

**`highlight`** highlights pixels near `None`, `Red`, `Green`, or `Blue`.

## Ambient light sensor (Windows only)

The `release/windows` branch supports a TCS3472 RGB ambient light sensor via
`rgb_sensor_print.ino.ino` (microcontroller firmware) and
`rgb_sensor_display.py` (host-side reader). The Linux deployment does not
include this.

## Training

Training code lives on the `training` branch.
