import shutil
from model import Net
from Library.spectrum_library import SpectrumCollection
import pandas as pd
import tensorflow as tf
import numpy as np
import chemparse
import pickle
import json
import os
import time
from multiprocessing import Pool

from class_model import Net as class_net
from matchms.filtering import interpret_pepmass, normalize_intensities
# from model_tensorflow import Net
from ms2deepscore import MS2DeepScore
from toolbox.ms2deepscore import SpectrumBinner
from ms2deepscore.vector_operations import cosine_similarity
from tqdm import tqdm as bar

from arg_parser import parse_args
from com_data_create import com_data_generate
from toolbox.MGF import MGF, write
from toolbox.mstool import (chunks,str2array, default_filters,normTanimotoDistance,num2array,checkFormulaSim)


from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse
from urllib.request import Request, urlopen
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
filterByMass = params.filter_mass  # 预处理中相对分子质量筛选阈值
filterByFormula = params.filter_formula  # 预处理中是否按照分子式筛选

outputNum = params.output_num  # 输出数量
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


c_classes = json.load(open("class_model/class/class_cf.json", "r"))
c_model = class_net(base_features_size=embedding_dim, output_size=len(
    c_classes), is_Multiclass=True)
c_model = c_model.cpu()
c_model.eval()
c_model.load_state_dict(torch.load("class_model/class/final.pth"))

sc_classes = json.load(open("class_model/superclass/superclass_cf.json", "r"))
sc_model = class_net(base_features_size=embedding_dim, output_size=len(
    sc_classes), is_Multiclass=True)
sc_model = sc_model.cpu()
sc_model.eval()
sc_model.load_state_dict(torch.load("class_model/superclass/final.pth"))

def argsort(seq, key, reverse=True):
    return sorted(range(len(seq)), key=lambda x: seq[x][key], reverse=reverse)



def processFpr(input):
    i, embeddings, pred_fpr, sourceSpec, mass = input
    res_list = []

    for embedding in embeddings:
        if filterByMass > 0 and abs(float(embedding[2]) - mass) >= filterByMass:
            continue
        if filterByFormula and not checkFormulaSim(chemparse.parse_formula(embedding[5]), chemparse.parse_formula(sourceSpec.get("formula"))):
            # print(embedding["formula"],chemparse.parse_formula(sourceSpec.get("formula")))
            continue
        # 14 bit fp    ; 10  MACCS
        sim = normTanimotoDistance(
            pred_fpr, str2array(embedding[14]))
        res = {
            # "sourceSpec": {
            #     "mass": sourceSpec.get("mass"),
            #     "file": sourceFile[1],
            #     "index": str(sourceFile[0]),
            # },
                "inchikey": embedding[1],
                "name": embedding[4],
                "mass": embedding[2],
                "formula": embedding[5],
                "smiles": embedding[6],
                "pred_sim": sim
        }
        res_list.append(res)
    return res_list

from fpr_model import DS_Net

ds_model = DS_Net(base_features_size=500, output_size=1410)
ds_model.load_state_dict(torch.load(ds_model_checkpoint_path + 'final.pth',map_location="cpu"))
ds_model.eval()
print("loading fpr database ...")
fpr_database = pd.read_csv(fpr_database_path)
list = fpr_database.values.tolist()

sim_res_top_map = []
class_res = []
count = 0

for sourceSpec in sourceSpectrums[:]:
    local_list = []
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
    c_res = c_model(embedding_vector).cpu().detach().numpy()
    max_index = np.argmax(c_res)
    c_pred = c_classes[max_index]
    
    sc_res = sc_model(embedding_vector).cpu().detach().numpy()
    max_index = np.argmax(sc_res)
    sc_pred = sc_classes[max_index]
    print("start matching similar molecules ...")
    if filterByFormula and (sourceSpec.get("formula") == None or len(sourceSpec.get("formula")) < 2):
        print("<- Warning -> filterByFormula has been activated and there are no formula fonund! Considering deactivate the option!")
        filterByFormula = False
    def sigmoid(z):
        z = np.array(z)
        return 1/(1 + np.exp(-z))
    pred_fpr = ds_model(embedding_vector)
    origin_pred_fpr = torch.sigmoid(pred_fpr.cpu().detach()).tolist()
    pred_fpr = [(1 if x > 0.5 else 0) for x in origin_pred_fpr]
    pbar = bar(total=len(list),mininterval=1)
    mass = float(sourceSpec.get("pepmass")[0])
    with Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(processFpr,
                                        [[i, embeddings, pred_fpr, sourceSpec, mass]
                                        for i, embeddings in enumerate(chunks(list, num_workers*5))]):
            if res is not None:
                local_list += res
                pbar.update(len(res))
    pbar.close()
    print("end ==> {} th of {} < processes {} samples >  ...sorting...".format(
        sourceFiles[count][0], sourceFiles[count][1], len(local_list)))
    sim_tops_ids = argsort(local_list, key="pred_sim", reverse=True)

    for i in range(len(sim_tops_ids)):
        local_list[sim_tops_ids[i]]["sim_rank"] = i+1
        local_list[sim_tops_ids[i]]["total"] = len(local_list)

    mass = float(sourceSpec.get("pepmass")[0])
    res_list = [local_list[i] for i in sim_tops_ids[:outputNum]]
    class_res.append({"precursor":mass,"file":sourceFiles[count][1],"index":str(sourceFiles[count][0]),"class_pred":c_pred,"superclass_pred":sc_pred})
    sim_res_top_map+=[{"precursor": mass,
            "file": sourceFiles[count][1],
            "pred_fpr":origin_pred_fpr,
            "index": str(sourceFiles[count][0]),
            **item
            } 
            for item in res_list]
    count += 1

if not os.path.exists("output/"):
    os.mkdir("output/")


import pandas as pd
from datetime import datetime
df = pd.DataFrame(sim_res_top_map)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df.to_csv('output/{}-similarity-matching.csv'.format(timestamp), index=False)

df = pd.DataFrame(class_res)
df.to_csv('output/{}-classification.csv'.format(timestamp), index=False)