# ComFaceID

ComFaceID is a tool for molecular similarity matching and classification based on spectral data. It processes input files, applies preprocessing steps, and uses machine learning models to predict molecular fingerprints and classify molecules into classes and superclasses.

---

## Parameters

The project uses the following input parameters, which can be configured via command-line arguments:

| Parameter                  | Type    | Default | Description                                                                 |
|----------------------------|---------|---------|-----------------------------------------------------------------------------|
| `--filter_mass`            | `float` | `200`   | Relative molecular mass screening thresholds in pretreatment.              |
| `--predict_fpr`            | `bool`  | `true`  | Use model output to predict fingerprint and compute similarity.            |
| `--filter_formula`         | `bool`  | `false` | Filter by molecular formula.                                               |
| `--inten_thresh`           | `float` | `1`     | Intensity threshold for preprocessing.                                     |
| `--rt`                     | `float` | `30`    | Retention time threshold.                                                  |
| `--ppm`                    | `float` | `20`    | PPM precision for matching signals between samples.                        |
| `--msdelta`                | `float` | `0.01`  | Msdelta threshold for merging spectra.                                     |
| `--if_merge_samples_byenergy` | `bool` | `false` | Whether to merge samples by energy.                                        |
| `--min_mz_num`             | `float` | `2`     | Minimum number of mz values required.                                      |
| `--remove_precursor`       | `bool`  | `true`  | Whether to remove precursor peaks during preprocessing.                    |
| `--output_num`             | `float` | `20`    | Number of most similar molecules to output.                                |

---

## Entry Function

The main entry point of the project is the 

main.py

 script. It initializes the parameters, loads input files, preprocesses spectral data, and performs molecular similarity matching and classification.

### Key Functions in 

main.py

:
- **`parse_args()`**: Parses input parameters from the command line.
- **`load_from_mgf(filename)`**: Loads spectral data from `.mgf` files.
- **`process(input)`**: Computes molecular similarity using embeddings.
- **`processFpr(input)`**: Computes molecular similarity using fingerprints.
- **`com_data_generate(specs)`**: Preprocesses spectral data.

---

## Usage Example

### Input Files

The project accepts input files in `.mzxml` or `.mgf` formats. These files should be placed in the `input/files` directory. 

If the input includes `.mzxml` files, a corresponding peak table in `.csv` format must also be placed in the `input/files` directory. The `.csv` file should contain the following columns:
- **`mz`**: Mass-to-charge ratio.
- **`rt`**: Retention time in seconds.

To run the project, use the following command:

```bash
python main.py --filter_mass 200 --predict_fpr true --filter_formula false --inten_thresh 1 --rt 30 --ppm 20 --msdelta 0.01 --if_merge_samples_byenergy false --min_mz_num 2 --remove_precursor true --output_num 20
```

This command processes spectral data with the specified parameters and outputs two CSV files:
1. **Similarity Matching Results**: `output/<timestamp>-similarity-matching.csv`
2. **Classification Results**: `output/<timestamp>-classification.csv`

---

## Required Files

The following files and folders are required for the project. Download them using the provided links and place them in the specified directories:

| Folder/File                | Description                                      | Download Link                                                                 |
|----------------------------|--------------------------------------------------|-------------------------------------------------------------------------------|
| 

base_model

               | Base model for predicting embeddings.           | [Download](https://zenodo.org/records/16676832?token=rK9LQv6ZaQld6cd639UQefIPpKusT2zdMvbeAIleanx0pzwQcqz1udyfP9WD4UVB9e_SCgW6ia3PgkpkGQikhw) |
| 

class_model/class

        | Model for class classification.                 | [Download](https://zenodo.org/records/16682538?token=1JIqruLgGaG0RMTAsVais6FPu9wABSAFmZNz3KYBy6mMALZ8P_-dBvfNEG8MddVhrGWRU80ULzFcpluDkTTscg) |
| 

class_model/superclass

   | Model for superclass classification.            | [Download](https://zenodo.org/records/16679031?token=6NigkHdDnp2lrDAufodaTYQoVROqmv_KB6L7EqWDYwElhySiID11sGChE31cMRiZAtpYJmF_d10gGYXowtkQpA) |
| 

fpr_model

                | Model for molecular fingerprint prediction.     | [Download](https://zenodo.org/records/16682503?token=ilkes2IK-5c3z75DfqVD_W0AbBWjA_gR3Rs6HwrqnnECJfDYQB0prznZQWAMoQzQV_i4WxChoGcoIFUyA7ostQ) |
| 

fpr_database

             | Database for molecular fingerprint similarity.  | [Download](https://zenodo.org/records/16679974?token=Kd8eYmSUSj2DB108pkOmikOstSpGTvsmo0S-CHoNyB37nP0nxx9IAepBRG8O62NRE9j3m0vwuvrwZuEpYb9KbA) |
| 


---

## Dependencies

The project requires the following Python libraries:
- `tensorflow`
- `torch`
- `numpy`
- `pandas`
- `matchms`
- `ms2deepscore`
- `chemparse`
- `argparse`
- `tqdm`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License.