# Seal Project

## 📺 [Tutorial Video For Using the App](https://youtu.be/Z47oREA_Vgk)

## Abstract
Identifying individual harbor seals is critical for marine mammal research and conservation. The Western Washington University Marine and Mammal Ecology Lab (MMEL) uses harbor seal identification and frequency tracking to manage salmon populations in local creeks. By tracking the frequency of specific seals, researchers determine which individuals are most significantly contributing to salmon predation. Using this information, they decide whether certain seal individuals need to be moved to another location to help preserve the salmon population. However, manual identification is time-consuming and prone to error. Our project addresses this challenge by developing a computer vision-based system to automate the identification of harbor seal individuals based on the unique patterns on their fur coat.
We explored various approaches, including normalization of sample counts per individual, image preprocessing techniques, and several convolutional neural network (CNN) architectures. After extensive experimentation, we selected Google’s pretrained Vision Transformer (ViT) model, which we fine-tuned on our dataset of labeled harbor seal images. The trained Vision Transformer model was then built into an app that the MMEL will be using to streamline the identification of new seal images. 
This system significantly improves the cataloging process, allowing lab members to spend less time manually labeling and more time analyzing results. The automation of identification not only increases efficiency but also enables quicker ecological insights and decision-making. Our work represents a meaningful step toward integrating machine learning into wildlife monitoring and conservation efforts.



## Installation

### Step 1: Download and Install Conda

If you don't have Conda installed, we recommend installing **Miniconda** (a minimal installer for Conda):

1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html)
2. Choose the installer for your operating system.
3. Follow the installation instructions provided on the site.

After installation, restart your terminal.

### 2: Clone the Repository
```bash
git clone https://github.com/seamus-parker/seal-identification.git
cd seal
````

### Step 3: Create the Conda Environment

This project includes a Conda environment file: environment.yaml.

To create the environment:

```bash
conda env create -f environment.yaml
```
> 📝 This will create a conda environment named `Seal` with all required dependencies.

Then activate the environment:

```bash
conda activate Seal
```

# Seal Individual Identifier: app.py

This Streamlit app allows users to upload images of harbor seals, crop them (optionally with OwlViT auto-crop assistance), and identify individuals using a fine-tuned ViT (Vision Transformer) classification model.


If you have not already, make sure to follow the miniconda installation at the beginning of the ReadMe and activate the conda env Seal this command
```bash
conda activate Seal
```

## Running the App

Once the environment is activated and folders are in place, run the following out while in the directory containing app.py:

```bash
streamlit run app.py
```

This will open the Seal Identifier interface in your default browser.

---

## Features

* Upload one or more images of seals.
* OwlViT auto-crops to detect the seal’s head (optional).
* Manual cropping supported via Streamlit Cropper.
* Predicts top 3 matching individuals with confidence.
* Save cropped images with custom names.
* Download all saved cropped images as a ZIP.

---

## Dependencies

All dependencies are handled by `environment.yaml`. These include:

* `streamlit`
* `streamlit-cropper`
* `torch`
* `transformers`
* `Pillow`
---



## Inference: infer.py

This script classifies images of seals using a fine-tuned Vision Transformer (ViT) model and saves the top-k predictions for each image to a CSV file.

If you have not already, make sure to follow the miniconda installation at the beginning of the ReadMe and activate the conda env Seal this command
```bash
conda activate Seal
```
### Requirements

Before running, make sure you have the following Python packages installed:

```bash
pip install transformers torch pandas pillow
```

### Usage

Run the script using the command line:

```bash
python infer.py -model_dir path/to/model -img_dir path/to/images -top_k 3 -output_file seal_predictions
```

### Arguments

| Argument       | Description                                                | Default       |
| -------------- | ---------------------------------------------------------- | ------------- |
| `-model_dir`   | Path to the directory containing your fine-tuned ViT model | `./model`     |
| `-img_dir`     | Directory containing input images to classify              | `./images`    |
| `-top_k`       | Number of top predictions to return per image              | `1`           |
| `-output_file` | Desired filename (without extension) for the output CSV    | `predictions` |

### Recommended File Structure
```
project-root/
│
├── infer.py                       # Inference script
│
├── model/                         # Fine-tuned ViT model directory
│   ├── config.json
│   ├── model.safetensors
│   ├── training_args.bin
│   └── preprocessor_config.json
│
├── images/                        # Folder with unclassified seal images
│   ├── seal1.jpg
│   ├── seal2.jpg
│   └── ...
│
└── predictions.csv                # Output CSV file (after running script)
```
## Notes:

* Place your trained ViT model and processor files in the `model/` folder (this is the default `-model_dir`).
* Place the seal images to classify in the `images/` folder (default `-img_dir`).
* The script will save predictions to `predictions.csv` (or your chosen `-output_file` name) in the root.


### Output

A CSV file (e.g., `seal_predictions.csv`) will be generated with the following format:

| filename    | label\_1   | score\_1 | label\_2   | score\_2 | ... |
| ----------- | ---------- | -------- | ---------- | -------- | --- |
| `seal1.jpg` | `Seal_023` | 0.9843   | `Seal_005` | 0.0132   |     |
| `seal2.png` | `Seal_005` | 0.9931   | `Seal_023` | 0.0049   |     |

Each row represents the predictions for a single image file.

### Example

```bash
python infer.py -model_dir ./saved_model -img_dir ./unseen_seals -top_k 2 -output_file results_0505
```
