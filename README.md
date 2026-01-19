# ComFaceID

[![Web Demo](https://img.shields.io/badge/Service-Web_Demo-blue)](https://npcompass.xulab.cloud/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

---

## 🌟 Core Functionalities

ComFaceID is a robust tool for molecular similarity matching and classification based on spectral data. It automates the pipeline from raw MS/MS spectra to chemical insight:

1. Spectral Similarity Matching: Utilizes embedding-based models to compute molecular similarity and retrieve analogous structures.
2. Molecular Fingerprint Prediction: Decodes spectral data to predict molecular fingerprints (--predict_fpr).
3. Hierarchical Classification: Automatically categorizes molecules into Classes and Superclasses using specialized deep learning models

---

## 📁 Resource Navigation

To maintain a lightweight repository, we separate logic from model weights. Please download the required resources and place them in the directories specified below:

| Folder/File               | Description                                     | Download Link |
|---------------------------|-------------------------------------------------|----------------|
| `base_model`              | Base model for predicting embeddings.          | [Download](https://zenodo.org/records/16676832) |
| `class_model/class`       | Model for class classification.                | [Download](https://zenodo.org/records/16739187) |
| `class_model/superclass`  | Model for superclass classification.           | [Download](https://zenodo.org/records/16739195) |
| `fpr_model`               | Model for molecular fingerprint prediction.    | [Download](https://zenodo.org/records/16682503) |
| `fpr_database`            | Database for molecular fingerprint similarity. | [Download](https://zenodo.org/records/16679974) |

---

## 🚀 Usage

### 1. Installation

```bash
git clone https://github.com/username/repo.git
cd repo
pip install -r requirements.txt
```

### 2. Data Preparation

Place your input files in the `input/files` directory.
*   **Supported Formats:** `.mgf` or `.mzxml`.
*   **Note for .mzxml:** A corresponding `.csv` peak table (columns: `mz`, `rt`) must also be present in the directory.

### 3. Execution

Run the main script with your desired parameters. Below is a standard usage example:

```bash
python main.py --filter_mass 200 \
               --predict_fpr true \
               --filter_formula false \
               --inten_thresh 1 \
               --rt 30 \
               --ppm 20 \
               --msdelta 0.01 \
               --if_merge_samples_byenergy false \
               --min_mz_num 2 \
               --remove_precursor true \
               --output_num 20
```

#### ⚙️ Configuration Parameters

You can customize the processing pipeline using the following command-line arguments:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--filter_mass` | `float` | `200` | Relative molecular mass screening thresholds for pretreatment. |
| `--predict_fpr` | `bool` | `true` | Enable fingerprint prediction and similarity computation. |
| `--filter_formula` | `bool` | `false` | Enable filtering by molecular formula. |
| `--inten_thresh` | `float` | `1` | Intensity threshold for noise removal during preprocessing. |
| `--rt` | `float` | `30` | Retention time threshold (seconds). |
| `--ppm` | `float` | `20` | PPM precision tolerance for matching signals. |
| `--msdelta` | `float` | `0.01` | Msdelta threshold for merging spectra. |
| `--if_merge_samples_byenergy`| `bool` | `false` | Whether to merge samples based on collision energy. |
| `--min_mz_num` | `float` | `2` | Minimum number of m/z values required per spectrum. |
| `--remove_precursor` | `bool` | `true` | Whether to remove precursor peaks during preprocessing. |
| `--output_num` | `float` | `20` | Number of most similar molecules to output in the results. |

### 4. Results

Outputs are generated in the `output/` folder with timestamped filenames:
*   `output/<timestamp>-similarity-matching.csv`
*   `output/<timestamp>-classification.csv`


## 🌐 Web Service

An online demo is available for quick validation of small batches:👉 Access [ComFaceID](https://npcompass.xulab.cloud/) (or use this [link](https://npcompass.zju.edu.cn/) for Mainland China).
