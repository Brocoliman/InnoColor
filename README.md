# InnoColor

InnoColor deployment for Ubuntu. PyTorch and TKinter.

```
innocolor/
│── main.py
│── settings.py: app and device (Linux and XBox) specifications
|── base/: parent objects for utils
|── utils/: interface objects and also app-specific processes for running models
|── interfaces/: script for the three TK interfaces
|── icons/
|
|── dsX.pth: datasets saved with information for each input-groundtruth pair along with model used
|
|── model/
|── checkpoint/
|── cvdsim/
|── sam2/
|── sam2_configs/
|
│── requirements.txt
│── README.md
```

## Installation

1. Install requirements
```bash
pip install -r requirements.txt
```
2. Download models (move these into their respective folders in `checkpoint/`) from [Google Drive](https://drive.google.com/drive/u/0/folders/1mCRWx0PGKODix-e9XUzvoWWF8T7DReiZ)
- `Dalt-NET` requires `classifier_190.pth` and `LUTs_190.pth`
- `InnoColor` requires `classifier_2000.pth`, `generator_200.pth`, `LUTs_2000.pth`, and `SAM2UNet-SOD.pth`
- `distill` requires `generator_100.pth`

3. Download sample datasets and move them into root folder
- `ds1.pth`
- `ds2.pth`
- `ds3.pth`

## Run
1. Change `FOLDER` in `settings.py` to the name of this project location
2. Run 
```bash
python main.py
```

## Quick Guide

Each interface (camera interface and data-editing interface) contains components of the InnoColor toolbar. This includes the `model`, `sim`, and `highlight` selection. 

You can choose between `model` types of 
- `None`: No Daltonization is applied
- `Hue Rotation` [P]: JB Huang's Hue Rotation matrix as described in [Huang et al., 2008](https://www.researchgate.net/publication/29622829_Enhancing_Color_Representation_for_the_Color_Vision_Impaired)
- `Dalt-NET` [P]: multi-mapping table based method described in [Wang, 2024](https://www.researchgate.net/publication/382209331_Creation_of_DALT-NET_Deep_Learning_3D_Lookup_Table_Generation_Approach_to_Daltonization_for_Dichromatic_Vision)
- `InnoColor` [P]: saliency based method described in [Wang, 2025](https://www.researchgate.net/publication/395368307_Innocolor_a_Pioneering_Image_Recoloring_Framework_Driven_by_Salient_Attention_Networks). Note that this is NOT the newest method
- `Distillation` [D]: newest method using data distillation into DDPA. Please read the [attached paper](https://drive.google.com/file/d/148Mjvrru1jd9AE15vewAnqz-BlWg_y-h/view?usp=sharing)
- Usually models developed for protanopia are fairly suitable for deuteranopia cases, and vice versa, although there are significant limitations.
- In the data-editing interface, models have an input parameter slider. No current model utilizes this yet, although this value is passed as a parameter

You can choose between `sim` types of
- `None`: no CVD
- `Protan`: protanopia or protanomaly, red cone deficiency, second most common case
- `Deutan`: deuteranopia or deuteranomaly, green cone deficiency, most common case
- `Tritan`: tritanopia or tritanomaly, blue cone deficiency, a much rarer case
- Both interfaces allow slider adjustment of the severity from 0 (no CVD) to 1 (-nopia, full blindness for respective cone)

You can `highlight` pixels of colors near
- `None`
- `Red`
- `Green`
- `Blue`

