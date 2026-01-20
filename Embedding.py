from model import Net
from library.spectrum_library import SpectrumCollection
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import os
import time

from class_model import Net as class_net
from matchms.filtering import interpret_pepmass, normalize_intensities
# from model_tensorflow import Net
from ms2deepscore import MS2DeepScore
from toolbox.ms2deepscore import SpectrumBinner

from arg_parser import parse_args
from com_data_create import com_data_generate
from toolbox.MGF import MGF
from toolbox.mstool import  default_filters

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

checkpoint_path =  "base_model/"
ds_model_checkpoint_path = "fpr_model/"
fpr_database_path = "fpr_database/CONAPUS_pubchem_export.csv"
input_peak_table = "input/peaktable/"
SourceDatabasePath = "input/files/"
TmpPath = "input/tmp_mgf/"

params = parse_args()
print("input params :\n", params)


os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


num_of_bin = 60000
embedding_dim = 500

extra_bit = 1
num_workers = 16  

SourcefileNameFilter = ""

if not os.path.exists(TmpPath):
    os.mkdir(TmpPath)


spectrum_binner = SpectrumBinner(
    num_of_bin, mz_min=0, mz_max=2000, peak_scaling=0.5, allowed_missing_percentage=100)
spectrum_binner = spectrum_binner.from_json(
    open(checkpoint_path + "spectrum_binner.json").read())
spectrum_binner.allowed_missing_percentage = 40

targetFile = []
targetSpectrums = []
sourceFiles = []
sourceSpectrums = []
fingerPrintMap = {}


def load_from_mgf(filename: str):
    for pyteomics_spectrum in MGF(filename, convert_arrays=1):
        if pyteomics_spectrum == None:
            continue
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        yield {"metadata": pyteomics_spectrum.get("params", None),
               "mz": mz,
               "intensities": intensities}

for file in os.listdir(SourceDatabasePath):
    peaktable = False
    if os.path.exists(input_peak_table):
        peaktable_list = os.listdir(input_peak_table)
        peaktable_list  = [i for i in peaktable_list if not i.endswith(".gitkeep")]
        peaktable = False if len(peaktable_list) == 0 else pd.read_csv(
            input_peak_table + peaktable_list[0])   
    if (file.lower().endswith(".mgf") or file.lower().endswith(".mzxml")):
        if SourcefileNameFilter != "" and SourcefileNameFilter not in file:
            continue
        origin = str(file)
        file = SourceDatabasePath + file
        query_collection = SpectrumCollection(str(file))
        query_collection.load_from_file(
            drop_mslevels=[1],
            inten_thresh=params.inten_thresh,
            remove_pre=params.remove_precursor,
            peaktable=peaktable,
            min_mz_num=params.min_mz_num,
            engine="utf-8")
        if len(query_collection.ms1list) == 0:
            print("WARNING!!! ms1list not found: ",file)
        else:
            query_collection.make_ms1list_sequential()
        query_collection.merge_neighbor_peaks()
        query_collection.merge_spectra(
            rt=params.rt, ms2delta=params.msdelta, by_col_e=params.if_merge_samples_byenergy)
        tmp_file = TmpPath + (origin.split(".")[0] + "__tmp__.mgf")
        with open(tmp_file, "w+") as file:
            query_collection.save_to_mgf(file)
        time.sleep(1)
        file = load_from_mgf(tmp_file)
        com_spectrum = []
        for i, spectrum in enumerate(file):
            np_sp,spectrum = com_data_generate(spectrum)
            spectrum = interpret_pepmass(spectrum)
            spectrum = default_filters(spectrum)
            # Scale peak intensities to maximum of 1
            spectrum = normalize_intensities(spectrum)
            sourceSpectrums.append(spectrum)
            com_spectrum.append(np_sp)
            sourceFiles.append((i, origin))
        os.remove(tmp_file)
        os.rmdir(TmpPath)
dimension = len(spectrum_binner.known_bins)

model = Net(spectrum_binner, base_features_size=dimension +
            extra_bit, training=False)
model.model.load_weights(str(checkpoint_path) + "/")
similarity_measure = MS2DeepScore(model)

def argsort(seq, key, reverse=True):
    return sorted(range(len(seq)), key=lambda x: seq[x][key], reverse=reverse)

count = 0
result = []

for sourceSpec in sourceSpectrums[:]:
    try:
        reference = spectrum_binner.transform([sourceSpec], progress_bar=False)[0]
    except Exception as e:
        print("unknown spec :", sourceFiles[i], " ",e)
        continue
    print("start {}/{} : {} th of {} ".format(
        count+1, len(sourceSpectrums), sourceFiles[count][0], sourceFiles[count][1]))
    vector = similarity_measure._create_input_vector(reference)
    vector[0][extra_bit:] = vector[0][0:dimension]
    vector[0][0] = spectrum._metadata["pepmass"][0]
    vector[0][1] = spectrum._metadata["collision_energy"]
    reference_vector = similarity_measure.model.base.predict([vector]
                                                             )[0]
    pub1, mac1 = None, None
    tf.stop_gradient(reference_vector)
    print("predicting class and super class ...")
    embedding_vector = torch.tensor(
            reference_vector, dtype=torch.float32).cpu()
    mass = float(sourceSpec.get("pepmass")[0])

    result+=[
        {
            "file": sourceFiles[count][1],
            "embedding": embedding_vector.numpy().tolist(),
            "mass": mass
        }
    ]
   
    count += 1

if not os.path.exists("output/"):
    os.mkdir("output/")

from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("output/{}-embedding_results.pkl".format(timestamp), "wb") as f:
    pickle.dump(result, f)
