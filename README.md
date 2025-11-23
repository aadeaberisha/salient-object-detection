# Salient Object Detection (SOD)

A complete **end-to-end Salient Object Detection system** built with **PyTorch**.  
It uses a custom **U-Net–inspired CNN encoder–decoder** to detect the most visually important object in an image  
and generate a binary saliency mask.

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment & Clone the Repository

```bash
# Clone the repository
git clone https://github.com/aadeaberisha/salient-object-detection.git
cd salient-object-detection

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- torch
- torchvision
- numpy
- pillow
- matplotlib
- tqdm
- ipywidgets

---

## 3. Dataset Setup

Before training, structure your dataset like this:

```
dataset_root/
 ├── train/
 │     ├── images/
 │     └── masks/
 ├── val/
 │     ├── images/
 │     └── masks/
 └── test/
       ├── images/
       └── masks/
```

The project uses the **ECSSD dataset**.  
Images/masks are automatically resized to **128×128**, normalized to **[0, 1]**, and converted to PyTorch tensors.

---

## 4. Data Loading & Augmentation

The `data_loader.py` module handles:

- Loading RGB images + grayscale masks

- Resizing to 128×128

- Normalization to [0,1]

- Augmentations:

- Horizontal flip

- Light rotation

- Safe random cropping

- Brightness/contrast jitter

- Masks are always thresholded to {0,1}.

---

## 5. Model Architecture (U-Net Style)

The model is a **lightweight U-Net** built from scratch in PyTorch (`sod_model.py`), designed for binary saliency mask prediction.

### 🔹 Encoder
- 4 levels  
- Each level: **Double Conv → BatchNorm → ReLU → Dropout (0.1)**  
- Followed by **MaxPooling(2×2)**  
- Channels: `32 → 64 → 128 → 256`

### 🔹 Bottleneck
- Double Conv block with **512 channels**  
- Captures high-level saliency features

### 🔹 Decoder
- 4 upsampling stages using **ConvTranspose2D**  
- **Skip connections** with encoder features  
- Double Conv blocks  
- Channels: `512 → 256 → 128 → 64 → 32`

### 🔹 Output
- `Conv2d(32, 1, kernel_size=1)`  
- `Sigmoid` → outputs a **1-channel mask** in `[0,1]`


### 🔹 Loss Function

```
0.7 * BCE + 0.3 * (1 - Soft IoU)
```

---

## 6. Run Training

To start training the SOD model:

```bash
python train.py
```

Training features:

- Adam optimizer (lr=5e-4)

- EarlyStopping (patience=5)

- Validation IoU per epoch

- Checkpoint saving (checkpoint.pth)

- Best-model saving (best_unet_improved.pth)

- Resume training if checkpoint exists-

### Checkpointing
The training script automatically saves:
- `checkpoint.pth` → model + optimizer + epoch (for resume)
- `best_unet_improved.pth` → best model based on validation loss

If a checkpoint exists, training resumes from the last saved epoch.

---

## 7. Run Evaluation

After training, evaluate the model on the test set:

```bash
python evaluate.py
```

Metrics computed:

- IoU
- Precision
- Recall
- F1-score
- MAE

Visualization includes for sample test images:

- Input image  
- Ground-truth mask  
- Predicted saliency mask  
- Overlay of prediction on top of the input image

---

## 8. Example Workflow

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

---

## 9. Design Decisions & Limitations

**Design Choices**
- Custom U-Net–style CNN built from scratch
- Combination loss: BCE + Soft IoU
- Data augmentation integrated directly in loader
- Clear separation of scripts
- Easy-to-run training and evaluation pipeline

**Limitations**
- Trained on a single dataset
- No GUI/web interface
- Training time depends on hardware

---

## 10. Demo (Interactive Visualization)

A simple interactive demo is included in `demo_notebook.ipynb`.

The demo allows you to:

- Upload any RGB image  
- Run the model in real time  
- View the **input image**, **predicted saliency mask**, and **overlay**  
- See the **inference time per image** (CPU/GPU)

This is useful for quick testing and for the project presentation.

---

## Author

**Adea Berisha**  
Xponian Program – AI Engineering Stream
