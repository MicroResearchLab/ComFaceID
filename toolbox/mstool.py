import json
import math
import os
import random
from tqdm import tqdm as bar
from matchms.filtering import normalize_intensities
# from toolbox.mstool import default_filters
# from toolbox.mstool import load_from_mgf
from toolbox.ms2deepscore import SpectrumBinner
from matchms.filtering.set_ionmode_na_when_missing import set_ionmode_na_when_missing
from matchms.filtering.make_ionmode_lowercase import make_ionmode_lowercase
from matchms.filtering.make_charge_int import make_charge_int
from matchms.filtering.interpret_pepmass import interpret_pepmass
from matchms.filtering.derive_ionmode import derive_ionmode
from matchms.filtering.derive_formula_from_name import derive_formula_from_name
from matchms.filtering.derive_adduct_from_name import derive_adduct_from_name
from matchms.filtering.correct_charge import correct_charge
from matchms.filtering.clean_compound_name import clean_compound_name
from matchms.filtering.add_precursor_mz import add_precursor_mz
from matchms.filtering.add_compound_name import add_compound_name
from typing import Generator, TextIO, Union
import numpy
from toolbox.MGF import MGF as MGF
from .Spectrum import Spectrum
from matchms.filtering import normalize_intensities, reduce_to_number_of_peaks
import logging
import numpy as np
from matchms.typing import SpectrumType
from matchms.Fragments import Fragments
from multiprocessing import Pool
import threading

random.seed(114514)

def list_split(full_list, ratio, shuffle=True):

    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_2,sublist_1



def remove_noise_peaks(spectrum_in: SpectrumType, noise_percentage=0) -> SpectrumType:

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) == 0:
        return spectrum

    assert numpy.max(
        spectrum.peaks.intensities) < 1.01, "you shall use remove_noise_peaks after calling normalize_intensities"

    remove_ids = np.where(spectrum.peaks.intensities <= (noise_percentage/100))
    mz = np.delete(spectrum.peaks.mz, remove_ids)
    intensities = np.delete(spectrum.peaks.intensities, remove_ids)
    spectrum.peaks = Fragments(mz=mz, intensities=intensities)

    return spectrum


# def merge_spec(spectrum_1: SpectrumType,spectrum_2:SpectrumType) -> SpectrumType:

#     if spectrum_in is None:
#         return None

#     spectrum = spectrum_in.clone()

#     if len(spectrum.peaks) == 0:
#         return spectrum

#     assert numpy.max(spectrum.peaks.intensities) !=1 , "you shall use remove_noise_peaks after calling normalize_intensities"

#     remove_ids = np.where(spectrum.peaks.intensities<=noise_percentage/100)[0]
#     spectrum.peaks.mz = np.delete(spectrum.peaks.mz,remove_ids)
#     spectrum.peaks.intensities = np.delete(spectrum.peaks.intensities,remove_ids)

#     return spectrum


def load_from_mgf(source: Union[str, TextIO], sortbyMZ=True,
                  metadata_harmonization: bool = True) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mgf file.

    This function will create ~matchms.Spectrum for every spectrum in the given
    .mgf file (or the file-like object).

    Examples:

    .. code-block:: python

        from matchms.importing import load_from_mgf

        file_mgf = "pesticides.mgf"
        spectra_from_path = list(load_from_mgf(file_mgf))

        # Or you can read the file in your application
        with open(file_mgf, 'r') as spectra_file:
            spectra_from_file = list(load_from_mgf(spectra_file))

    Parameters
    ----------
    source:
        Accepts both filename (with path) for .mgf file or a file-like
        object from a preloaded MGF file.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """

    for pyteomics_spectrum in MGF(source, convert_arrays=1):

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]
        if sortbyMZ:
            idx_sorted = numpy.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]
        # if not sortbyMZ:
        #     idx_sorted = numpy.argsort(intensities)
        #     mz = mz[idx_sorted]
        #     intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz,
                       intensities=intensities,
                       
                       metadata=metadata,
                       metadata_harmonization=metadata_harmonization)





def default_filters(spectrum: SpectrumType) -> SpectrumType:
    """
    Collection of filters that are considered default and that do no require any (factory) arguments.

    Collection is

    1. :meth:`~matchms.filtering.make_charge_int`
    2. :meth:`~matchms.filtering.make_ionmode_lowercase`
    3. :meth:`~matchms.filtering.set_ionmode_na_when_missing`
    4. :meth:`~matchms.filtering.add_compound_name`
    5. :meth:`~matchms.filtering.derive_adduct_from_name`
    6. :meth:`~matchms.filtering.derive_formula_from_name`
    7. :meth:`~matchms.filtering.clean_compound_name`
    8. :meth:`~matchms.filtering.interpret_pepmass`
    9. :meth:`~matchms.filtering.add_precursor_mz`
    10. :meth:`~matchms.filtering.derive_ionmode`
    11. :meth:`~matchms.filtering.correct_charge`

    """
    # spectrum = make_charge_int(spectrum)
    spectrum = make_ionmode_lowercase(spectrum)
    spectrum = set_ionmode_na_when_missing(spectrum)
    # spectrum = add_compound_name(spectrum)
    # spectrum = derive_adduct_from_name(spectrum)
    # spectrum = derive_formula_from_name(spectrum)
    # spectrum = clean_compound_name(spectrum)
    # spectrum = interpret_pepmass(spectrum)
    spectrum = add_precursor_mz(spectrum)
    # spectrum = derive_ionmode(spectrum)
    # spectrum = correct_charge(spectrum)
    return spectrum

def processFile(input):
    file, DatabasePath, noise_percentage = input
    if file.endswith(".mgf") and not file.endswith(".ni.mgf") and not file.endswith(".del"):
        file = os.path.join(DatabasePath, file)
    else : 
        return None
    file = load_from_mgf(file)
    out = []
    for spectrum in file:
        spectrum = default_filters(spectrum)
        # Scale peak intensities to maximum of 1
        spectrum = normalize_intensities(spectrum)
        # if noise_percentage > 0:
        #     spectrum = remove_noise_peaks(
        #         spectrum, noise_percentage=noise_percentage)
        spectrum = reduce_to_number_of_peaks(
            spectrum, n_required=8, n_max=10000)
        if spectrum is not None:
            out.append(spectrum)
    return out



def getSpecLoader(DatabasePath, test_DatabasePath, test_mode=False, noise_percentage=1, number_of_bins=4500,num_workers=16):
    spectrum_binner = SpectrumBinner(
        number_of_bins, mz_min=0, mz_max=2000, peak_scaling=0.5)
    spectrums_train = []
    spectrums_test = []
    if test_mode:
        files = os.listdir(DatabasePath)[:500]
    else:
        files = os.listdir(DatabasePath)
    print("==> load train dataset [total: {:d}]:".format(len(files)))
    pbar = bar(total=len(files))
    with Pool(processes=num_workers) as pool:
        for spec in pool.imap_unordered(processFile,
        [[file,DatabasePath, noise_percentage] for file in files]):
            if spec is not None:
                spectrums_train+=spec
            pbar.update(1)
    print("    {:d} files loaded".format(len(spectrums_train)))

    print("load test dataset:")
    if test_mode:
        files = os.listdir(test_DatabasePath)[:500]
    else:
        files = os.listdir(test_DatabasePath)

    print("==> load test dataset [total: {:d}]:".format(len(files)))
    pbar = bar(total=len(files))
    with Pool(processes=num_workers) as pool:
        for spec in pool.imap_unordered(processFile,
        [[file,test_DatabasePath, noise_percentage] for file in files]):
            if spec is not None:
                spectrums_test+=spec
            pbar.update(1)
    print("    {:d} files loaded".format(len(spectrums_test)))
    spectrum_binner.fit_transform(spectrums_train+spectrums_test)
    binned_spectrums = spectrum_binner.transform(spectrums_train)
    binned_spectrums_test = spectrum_binner.transform(spectrums_test)

    return spectrum_binner, binned_spectrums, binned_spectrums_test


def processFileForClassifier(input):
    file, LabelPath, suffix, class_key, classes, noise_percentage,make_single_class = input
    if file.endswith(".mgf") and not file.endswith(".ni.mgf") and not file.endswith(".del"):
        file = load_from_mgf(file)
        # labelPath = os.path.join(LabelPath, file.split("/")[-1].split("_")[
        #                          0]+"-N_NP"+suffix+".json")
        labels = None
        out = []
        for spectrum in file:
            if labels ==None:
                labelPath = os.path.join(LabelPath, spectrum.get("inchikey",None)+".json")
                if not os.path.isfile(labelPath):
                    labelPath = os.path.join(LabelPath, spectrum.get("inchikey",None)+suffix+".json")
                    if not os.path.isfile(labelPath):
                        print("unknown label inchikey: "+ spectrum.get("inchikey",None))
                        continue
                labelFile = json.load(open(labelPath, "r"))
                if class_key in labelFile:
                    labelFile = labelFile[class_key]
                    if len(labelFile) == 0:
                        return None
                else:
                    return None
                labels = [0 for _ in range(len(classes))]
                if make_single_class:
                    for i in range(len(classes)):
                        if classes[i] == str(labelFile):
                            labels[i] = 1
                else:
                    for i in range(len(classes)):
                        if classes[i] in labelFile:
                            labels[i] = 1
            spectrum = default_filters(spectrum)
            # Scale peak intensities to maximum of 1
            spectrum = normalize_intensities(spectrum)
            if noise_percentage > 0:
                spectrum = remove_noise_peaks(
                    spectrum, noise_percentage=noise_percentage)
            spectrum = reduce_to_number_of_peaks(
                spectrum, n_required=8, n_max=10000)
            if spectrum is not None:
                spectrum.set("label", labels)
                out.append(spectrum) 
        return out


def getSpecLoaderForClassifier(DatabasePath, test_DatabasePath, LabelPath, class_key, ClassesPath,
                                noise_percentage=1, number_of_bins=4500, num_workers=16,re_shuffle=True,make_single_class=False):
    spectrum_binner = SpectrumBinner(
        number_of_bins, mz_min=0, mz_max=2000, peak_scaling=0.5)
    spectrums_train = []
    spectrums_test = []
    if "/CF" in LabelPath:
        suffix = "_CF"
    if "/NP" in LabelPath:
        suffix = "_NP"
    classes = json.load(open(ClassesPath, "r"))
    
    l1 = [os.path.join(DatabasePath, file) for file in  os.listdir(DatabasePath)]
    l2 = [os.path.join(test_DatabasePath, file) for file in  os.listdir(test_DatabasePath)]
    if re_shuffle:
        files,test_files = list_split(l1+l2,0.2)
    else:
        files = l1
        test_files = l2
    print("==> load train dataset [total: {:d}]:".format(len(files)))
    pbar = bar(total=len(files))
    with Pool(processes=num_workers) as pool:
        for spec in pool.imap_unordered(processFileForClassifier,
        [[file, LabelPath, suffix, class_key, classes, noise_percentage,make_single_class] for file in files]):
            if spec is not None:
                spectrums_train+=spec
            pbar.update(1)
    print("    {:d} files loaded".format(len(spectrums_train)))


    

    print("==> load test dataset [total: {:d}]:".format(len(test_files)))
    pbar = bar(total=len(test_files))
    with Pool(processes=num_workers) as pool:
        for spec in pool.imap_unordered(processFileForClassifier,
        [[file, LabelPath, suffix, class_key, classes, noise_percentage,make_single_class] for file in test_files]):
            if spec is not None:
                spectrums_test+=spec
            pbar.update(1)
    print("    {:d} files loaded".format(len(spectrums_test)))
    spectrum_binner.fit_transform(spectrums_train+spectrums_test)
    binned_spectrums = spectrum_binner.transform(spectrums_train)
    binned_spectrums_test = spectrum_binner.transform(spectrums_test)

    return spectrum_binner, binned_spectrums, binned_spectrums_test

def get_file_list(path,list=[]):
    if os.path.isfile(path): 
        list.append(path)
    elif os.path.isdir(path):
        for s in os.listdir(path):
            newPath = os.path.join(path,s)
            get_file_list(newPath,list)
    return list



def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]




def normTanimotoDistance(vec1, vec2):
    intersection = np.multiply(vec1, vec2)  # 利用预测值与标签相乘当作交集
    union = np.bitwise_or(vec1, vec2)
    score = (np.sum(intersection) + 1e-5) / (np.sum(union) + 1e-5)
    return score

def num2array(din, bit_width, padding=0):
    bin_obj = bin(int(din))[(2+padding):]
    bin_str = bin_obj.rjust(bit_width, '0')  # 高位补0
    o_arr = []
    for ii in range(len(bin_str)):
        o_arr.append(int(bin_str[len(bin_str)-ii-1]))
    return o_arr

def num2binstr(din, bit_width, padding=0):
    bin_obj = bin(int(din))[(2+padding):]
    bin_str = bin_obj.rjust(bit_width, '0')  # 高位补0
    o_arr = ""
    for ii in range(len(bin_str)):
        o_arr+=str(int(bin_str[len(bin_str)-ii-1]))
    return o_arr
def str2array(in_str):
    res = []
    for s in in_str:
        res.append(int(s))
    return np.array(res) 


def checkFormulaSim(formulaA: dict, formulaB: dict):
    # formulaA.setdefault("N",0)
    # formulaB.setdefault("N",0)
    # formulaA.setdefault("O",0)
    # formulaB.setdefault("O",0)
    # formulaA.setdefault("S",-1)
    # formulaB.setdefault("S",-1)
    # formulaA.setdefault("Cl",-1)
    # formulaB.setdefault("Cl",-1)
    # formulaA.setdefault("P",-1)
    # formulaB.setdefault("P",-1)
    # if abs(formulaA["N"] - formulaB["N"])>5:
    #     return False
    # if abs(formulaA["O"] - formulaB["O"])>5:
    #     return False
    # if abs(formulaA["P"] - formulaB["P"])>2:
    #     return False
    # if formulaA["P"]*formulaB["P"] <0 or abs(formulaA["P"] - formulaB["P"])>2:
    #     return False
    # if formulaA["S"]*formulaB["S"] <0 or abs(formulaA["S"] - formulaB["S"])>5:
    #     return False
    # if formulaA["Cl"]*formulaB["Cl"] <0 or abs(formulaA["Cl"] - formulaB["Cl"])>5:
    #     return False
    # for key in {**formulaA, **formulaB}:
    #     if key in ["C","H","O","S","Cl","P","+","-"]:
    #         continue
    #     formulaA.setdefault(key,0)
    #     formulaB.setdefault(key,0)
    #     if formulaA[key] !=formulaB[key]:
    #         return False
    # return True
    return str(formulaA) == str(formulaB)