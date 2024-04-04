# **SAMCL Inference**
This code supports only the inference with pretrained SAM-CL model. Please visit [this repository](https://github.com/PhysiologicAILab/SAM-CL) for training code.

### **Installation**
* Clone or download the repo.
* Activate python virtual environment (venv or conda). Create virtual environment if it not available.
* Install the required Python packages with following command:
```bash
cd <path to SAMCL_Inference repo>
pip install -r requirements.txt
```

#### **Checkpoint Download**
Download the [checkpoint from this link](https://drive.google.com/drive/folders/1durAP--yz51W9WAKdTZ7XgIUpSngKrxA?usp=share_link) and place it under *"ckpt"* folder.

### **Usage Instructions:**

#### Terminal command to run inference on stored raw thermal images:
```bash
python inference.py --datadir *<path containing .bin raw thermal file>* --outdir *<path to store segmentation masks and visualization images>* --config *<config file with input parameters>* [--gpu *gpu_number*]
```

#### Illustrative terminal commands to run inference on raw images
**Inference with CPU:**
```bash
python inference.py --datadir data/test/ --outdir ./out/test --config configs/AU_SAMCL.json
```
**Inference with single GPU:**
```bash
python inference.py --datadir data/test/ --outdir ./out/test --config configs/AU_SAMCL.json --gpu 0
```
#### **Output**
There are two outcomes that code generates in the specified *outdir* path:
1. **PNG files** having segmentation masks, with semantic regions coded as follows:
   ```
    0 - background
    1 - cheek
    2 - mouth
    3 - eyes
    4 - eyebrows
    5 - nose.
   ```
2. **JPG files** (in a sub-folder with name "vis") showing overlaid visualization as the image below. These cannot be used for further processing.

#### :scroll: Citation
```
@inproceedings{Joshi_2022_BMVC,
    author={Jitesh N Joshi and Nadia Berthouze and Youngjun Cho},
    title={Self-adversarial Multi-scale Contrastive Learning for Semantic Segmentation of Thermal Facial Images},
    booktitle={33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
    publisher={{BMVA} Press},
    year={2022},
    url={https://bmvc2022.mpi-inf.mpg.de/0864.pdf}
}
```
