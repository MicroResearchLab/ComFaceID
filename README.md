# ComFaceID

[![Web Demo](https://img.shields.io/badge/Service-Web_Demo-blue)](https://npcompass.xulab.cloud/)

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
---

## 🌟 Core Functionalities

ComFaceID is a foundation model that supports diverse downstream tasks:

1. **Spectral Library Search**: Maps MS² spectra into 500-dimensional embeddings for rapid similarity search.
2. **Structural Classification**: Automatically categorizes compounds into chemical classes and superclasses.
3. **Molecular Fingerprint Prediction**: Decodes embeddings to predict molecular fingerprints.
4. **Compound Library Retrieval**: Retrieves analogous structures by querying databases with the predicted molecular fingerprints.
5. **MSNs**: A multi-parameter framework that integrates prediction metrics to assess the structural novelty of unknown compounds and prioritize "dark matter" for isolation.

---

## 📁 Resource Navigation

Please download the required resources and place them in the directories specified below:

| Folder/File | Description | Download Link |
| --- | --- | --- |
| `base_model` | Base model for predicting embeddings. | [Download](https://zenodo.org/records/16676832) |
| `class_model/class` | Model for class prediction. | [Download](https://zenodo.org/records/16739187) |
| `class_model/superclass` | Model for superclass prediction. | [Download](https://zenodo.org/records/16739195) |
| `fpr_model` | Model for molecular fingerprint prediction. | [Download](https://zenodo.org/records/16682503) |
| `fpr_database` | Database for molecular fingerprint based library retrieval. | [Download](https://zenodo.org/records/16679974) |

---

## 🚀 Usage

### 1. Installation

```bash
git clone <https://github.com/MicroResearchLab/ComFaceID.git>
cd ComFaceID
pip install -r requirements.txt

```

### 2. Data Preparation

Place your input files in the `input/files` directory.

- **Supported Formats:** `.mgf` or `.mzxml`.
- **Note for .mzxml:** A corresponding `.csv` peak table (columns: `mz`, `rt`) must also be present in the directory.

### 3. Embedding Generation (Standalone)

If you only need to convert mass spectrometry files into spectral embeddings (saved in **Pickle `.pkl`** format), use the `Embedding.py` script.

**Note:** This script supports the **exact same command-line arguments** as `main.py` (see the *Configuration Parameters* table below). This allows you to apply the same preprocessing, filtering, and merging logic during embedding generation.

```bash
python Embedding.py  --inten_thresh 1 \
               --rt 30 \
               --ppm 20 \
               --msdelta 0.01 \
               --if_merge_samples_byenergy false \
               --min_mz_num 2 \
               --remove_precursor true

```

**Output:** The generated embeddings will be saved in **Pickle (`.pkl`)** format.

### 4. Main Analysis (Full Pipeline)

Run the main script with your desired parameters. Below is a standard usage example:

```bash
python main.py --filter_mass 2000 \
               --filter_formula false \
               --inten_thresh 1 \
               --rt 30 \
               --ppm 20 \
               --msdelta 0.01 \\
               --if_merge_samples_byenergy false \
               --min_mz_num 2 \
               --remove_precursor true \
               --output_num 200

```

### ⚙️ Configuration Parameters

You can customize the processing pipeline using the following command-line arguments:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--filter_mass` | `float` | `2000` | Relative molecular mass screening thresholds for pretreatment. |
| `--inten_thresh` | `float` | `1` | Intensity threshold for noise removal during preprocessing. |
| `--rt` | `float` | `30` | Retention time threshold for feature alignment (seconds). |
| `--ppm` | `float` | `20` | ppm precision tolerance for feature alignment. |
| `--msdelta` | `float` | `0.01` | msdelta threshold for merging spectra. |
| `--if_merge_samples_byenergy` | `bool` | `false` | merge spectra by different collision energy or merge all. |
| `--min_mz_num` | `float` | `2` | Minimum number of fragments required per spectrum. |
| `--remove_precursor` | `bool` | `true` | remove precursor ions during preprocessing. |
| `--output_num` | `float` | `200` | Number of most similar molecules to output in the results. |

### 5. Results

Outputs from `main.py` are generated in the `output/` folder with timestamped filenames:

- `output/<timestamp>-similarity-matching.csv`
- `output/<timestamp>-classification.csv`

### 6. Metabolite Structural Novelty Score (MSNs)

After generating the similarity and classification results via `main.py`, you can utilize the **Metabolite Structural Novelty Score (MSNs)** to identify potentially novel compounds.

**Command:**

```bash
python MSNs.py \
  --sim_score_file_path output/<timestamp>-similarity-matching.csv \
  --class_results_file_path output/<timestamp>-classification.csv

```

> Note: Replace <timestamp> with the actual timestamp string found in your output/ directory filenames.
> 

**Parameters:**

| Parameter | Description |
| --- | --- |
| `--sim_score_file_path` | Path to the **Compound Library Retrieval** results CSV file generated by `main.py`. |
| `--class_results_file_path` | Path to the **Structural Classification** result Json file generated by `main.py`. |

## 🌐 Web Service

An online demo is available for quick validation of small batches:👉  [Access ComFaceID](https://npcompass.xulab.cloud/) (or use this [link](https://npcompass.zju.edu.cn/) for Mainland China).