import binascii
import struct
import zlib
from Library import numerical_utilities as numerical_utilities
from collections import defaultdict
import pandas as pd
import numpy as np
import os

element_mass_dict = {'H':1.007825,
                     'Na':22.98977,
                     'H2O':18.010565,
                     'H-H2O':17.00274,
                     'K':38.963706,
                     'Cl':34.968853,
                     'NH4':18.034374,
                     'H-Cl':33.961028,
                     'C2H3O2':59.013305}
# [('-', 'H'), ('+', 'H'), ('+', 'H-H2O'), ('+', 'Cl'), ('+', 'Na'), ('+', 'NH4'), ('+', 'H-Cl'), ('-', 'C2H3O2'), (']', '')]
# pl1 and pl2 are lists containing tuples
def merge_peaklist(pl1, pl2, ms2delta=0.01):
    # this method sort all the peaks first so we don't need to feed in the order peak list
    len1 = len(pl1)
    if len1 < 1:
        return pl2
    len2 = len(pl2)
    if len2 < 1:
        return pl1
    # TODO: no need to use pandas transform to sort!!!!!
    pl_new = pd.concat([pd.DataFrame(pl1), pd.DataFrame(pl2)], axis=0)
    pl_new = pl_new.sort_values(pl_new.columns[0])
    pl_new.reset_index(drop=True,inplace=True)
    i = 0; len_new = len(pl_new)
    while i < (len_new - 1):
        mz1 = pl_new.iloc[i,0]
        mz2 = pl_new.iloc[i+1,0]
        if abs(mz1 - mz2) < ms2delta:
            inten1 = pl_new.iloc[i,1]
            inten2 = pl_new.iloc[i+1,1]
            mz_new = round((mz1 + mz2)/2, 4)
            inten_new = round(inten1 + inten2, 2)
            pl_new.iloc[i, 0] = mz_new
            pl_new.iloc[i,1] = inten_new
            pl_new.drop(pl_new.index[i+1],inplace=True)
            len_new = len_new - 1
        else:
            i = i + 1
    # convert dataframe to list containing tuples

    return pl_new.values

def merge_peaklists(pls, ms2delta=.01):
    pl_all = np.concatenate(pls, axis=0)


#Decode peaks for mzXML
def decode_spectrum(line, peaks_precision, peaks_compression, struct_iter_ok):

    """https://groups.google.com/forum/#!topic/spctools-discuss/qK_QThoEzeQ"""

    decoded = binascii.a2b_base64(line)
    number_of_peaks = 0
    unpack_format1 = ""


    if peaks_compression == "zlib":
        decoded = zlib.decompress(decoded)

    #Assuming no compression
    if peaks_precision == 32:
        number_of_peaks = len(decoded)/4
        unpack_format1 = ">%df" % number_of_peaks
    else:
        number_of_peaks = len(decoded)/8
        unpack_format1 = ">%dd" % number_of_peaks

    # peaks = []
    # if struct_iter_ok:
    #     peak_iter = struct.iter_unpack(unpack_format1,decoded)
    #     peaks = [
    #        pair for pair in zip(*[peak_iter] * 2)
    #     ]
    # else:
    peaks = [
       [pair[0], pair[1]] for pair in zip(*[iter(struct.unpack(unpack_format1,decoded))] * 2)
    ]
    return peaks
    # peaks_list = struct.unpack(unpack_format1,decoded)
    # return [
    #     (peaks_list[i*2],peaks_list[i*2+1])
    #     for i in range(0,int(len(peaks_list)/2))
    # ]


def filter_precursor_peaks(peaks, tolerance_to_precursor, mz):
    new_peaks = []
    for peak in peaks:
        if abs(peak[0] - mz) > tolerance_to_precursor:
            new_peaks.append(peak)
    return new_peaks


def filter_noise_peaks(peaks, min_snr):
    average_noise_level = numerical_utilities.calculate_noise_level_in_peaks(peaks)
    new_peaks = []
    for peak in peaks:
        if peak[1] > average_noise_level * 10:
            new_peaks.append(peak)
    return new_peaks


def filter_peaks_noise_or_window(peaks, min_snr, window_size, top_peaks):
    window_filtered_peaks = window_filter_peaks(peaks, window_size, top_peaks)
    snr_peaks = filter_noise_peaks(peaks, min_snr)

    peak_masses_to_keep = []
    for peak in window_filtered_peaks:
        peak_masses_to_keep.append(peak[0])
    for peak in snr_peaks:
        peak_masses_to_keep.append(peak[0])

    peak_masses_to_keep = set(peak_masses_to_keep)

    new_peak = []
    for peak in peaks:
        if peak[0] in peak_masses_to_keep:
            new_peak.append(peak)

    return new_peak


def filter_to_top_peaks(peaks, top_k_peaks):
    sorted_peaks = sorted(peaks, key=lambda peak: peak[1], reverse=True)
    return sorted_peaks[:top_k_peaks]


def window_filter_peaks(peaks, window_size, top_peaks):
    peak_list_window_map = defaultdict(list)
    for peak in peaks:
        mass = peak[0]
        mass_bucket = int(mass/window_size)
        peak_list_window_map[mass_bucket].append(peak)

    new_peaks = []
    for bucket in peak_list_window_map:
        peaks_sorted_by_intensity = sorted(peak_list_window_map[bucket], key=lambda peak: peak[1], reverse=True)
        peaks_to_keep = peaks_sorted_by_intensity[:top_peaks]
        new_peaks += peaks_to_keep

    new_peaks = sorted(new_peaks, key=lambda peak: peak[0])
    return new_peaks


def export_odd_neutral_loss(db):
    ls = []
    for spec_c in db.SpectrumCollection_list:
        for spec in spec_c.spectrum_list:
            a = np.argwhere(spec.neutral_loss[:, 0] < -5)
            if len(a) != 0:
                ls.append((spec.filename, np.min(spec.neutral_loss[:, 0])))
    file_nm = open('MID_list.txt', 'w')
    file_min = open('MID_min.txt', 'w')
    for (nm, min_v) in ls:
        file_nm.write(os.path.basename(nm) + '\n')
        file_min.write(str(min_v) + '\n')

def export_ls2txt(data, output_file):
    file_nm = open(output_file, 'w')
    for value in data:
        try:
            total = ''
            for i in value:
                total = total + str(i) + '  '
        except:
            total = str(value)

        file_nm.write(total + '\n')

