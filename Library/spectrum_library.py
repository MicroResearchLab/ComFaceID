#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import Library.spectrum_alignment as spectrum_alignment
import Library.numerical_utilities as numerical_utilities
from Library import psm_library
import pandas as pd
import re
import time
import xmltodict
import numpy as np
import concurrent.futures
# from Library.ming_spectrum_library import Spectrum
from Library.utils import merge_peaklist, window_filter_peaks, filter_precursor_peaks, decode_spectrum, \
    element_mass_dict
import multiprocessing
# subprocess
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# import pyqtgraph as pg
# import pyqtgraph.opengl as gl


def load_frag_db(foldername, peaktable, drop_mslevels, inten_thresh, engine, index_low, index_up, remove_pre,
                 min_mz_num, processor_num):
    """
    Called in Database.load_file, this method implements the mono-processor task
    load spectra from files
    """
    num = 1
    spec_c_ls = []
    MID_ls = []
    for file in os.listdir(foldername)[index_low:index_up]:
        print(file)
        if file.lower().endswith(".mgf") or file.lower().endswith(".mzxml"):
            filename = os.path.join(foldername, file)
            SpecCol = SpectrumCollection(filename)
            SpecCol.load_from_file(peaktable=peaktable, drop_mslevels=drop_mslevels, inten_thresh=inten_thresh,
                                   engine=engine, remove_pre=remove_pre, min_mz_num=min_mz_num)
            if num % 100 == 0:
                print('\rprocessor {}, {} files loaded'.format(
                    processor_num, num), end='', flush=True)
            num += 1
            if len(SpecCol.spectrum_list) > 0:
                spec_c_ls.append(SpecCol)
                if (hasattr(SpecCol, 'MID')):
                    MID_ls.append(SpecCol.MID)
                else:
                    MID_ls.append('None')
        else:
            print('only mzxml and mgf files are supported')

    return spec_c_ls, MID_ls


class Database:
    # Database class use decompose_with_ions_in mgf() to save data to .mgf files
    def __init__(self, foldername, DBname):
        self.foldername = foldername
        self.name = DBname
        self.SpectrumCollection_list = []
        # IDlist contains the ID, mass, precursor info for all compound in the database
        # self.IDlist = pd.DataFrame(columns={'ID','precursor','mass'})
        self.loss_feature = None
        self.frag_feature = None
        self.MID_ls = []
        self.total_ion_pairs = []

    def load_files(self, peaktable=False, drop_mslevels=[1], inten_thresh=0, engine='utf-8', n_workers=16, min_mz_num=1,
                   remove_pre=False):
        """
        Load all the files in self.folders
        :param inten_thresh: the min percent of intensity to remain
        :param engine: the encoder engine to use (GBK, utf-8, ...)
        :param n_workers: how many process to use
        :return:
        """
        print('-----------------------------------------------------------------------')
        print('Start loading files in databse located at {}'.format(self.foldername))
        nm_ls = []
        for f in os.listdir(self.foldername):
            if f.endswith('.mgf') and not f.endswith(".ni.mgf"):
                nm_ls.append(f)
        file_num = len(nm_ls)
        n_cpu = n_workers
        pool = multiprocessing.Pool(processes=None)
        Ms = []
        base = int(file_num / n_cpu)
        if file_num < n_cpu:
            for i in range(file_num):
                Ms.append(pool.apply_async(load_frag_db, (self.foldername, peaktable, drop_mslevels, inten_thresh,
                                                          engine, i, i + 1, remove_pre, min_mz_num,
                                                          i + 1)))
        else:
            for i in range(n_cpu):
                if i + 1 < n_cpu:
                    Ms.append(pool.apply_async(load_frag_db, (self.foldername, peaktable, drop_mslevels, inten_thresh,
                                                              engine, i *
                                                              base, (i + 1) *
                                                              base, remove_pre, min_mz_num,
                                                              i + 1)))
                else:
                    Ms.append(pool.apply_async(load_frag_db, (self.foldername, peaktable, drop_mslevels, inten_thresh,
                                                              engine, i * base, file_num, remove_pre, min_mz_num,
                                                              i + 1)))
        pool.close()
        pool.join()
        Ms = [M.get() for M in Ms]
        for spec_c_ls, MID_ls in Ms:
            self.SpectrumCollection_list.extend(spec_c_ls)
            self.MID_ls.extend(MID_ls)
        #
        for spec_c in self.SpectrumCollection_list:
            if hasattr(spec_c, 'ion_list'):
                for pair in spec_c.ion_list:
                    if pair not in self.total_ion_pairs:
                        self.total_ion_pairs.append(pair)
        print('')
        print('Loading files finished, total file number:{}'.format(file_num))
        print('-----------------------------------------------------------------------')

    def filter_noise_peaks(self, min_snr):
        for spec_c in self.SpectrumCollection_list:
            for spec in spec_c.spectrum_list:
                spec.filter_noise_peaks(min_snr)

    def search_by_MID(self, MID):
        """
        Return a list containing all the matched SpectrumCollection in self.SpectrumCollection_list
        does not change the contents of self
        :param MID: desired MID
        :return: List of SpectrumCollection
        """
        if len(self.SpectrumCollection_list) == 0:
            print('load file first')
            return
        rs = []
        for spec_c in self.SpectrumCollection_list:
            if spec_c.MID == MID:
                rs.append(spec_c)
        return rs

    def get_MID(self, MID):
        # TODO: remove this function, same as search_by_MID
        new_ls = []
        for spec_c in self.SpectrumCollection_list:
            if spec_c.MID == MID:
                new_ls.append(spec_c)
        return new_ls

    # merge spectra from the same compound and same precursor
    def merge_spectra(self, ms2delta=0.01, ppm=20, rt=30, by_col_e=True):
        """
        notice that using the decomposed database is required
        , which means it's meaningless if you merge a non-decomposed database
        """
        if len(self.SpectrumCollection_list) == 0:
            print('load files first')
            return
        #  count of the SpectrumCollection numbers processed
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for spec_coll in self.SpectrumCollection_list:
                futures.append(executor.submit(
                     spec_coll.merge_spectra, ms2delta=ms2delta, ppm=ppm, rt=rt, by_col_e=by_col_e))
            count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if count % 100 == 0:
                        print('\rNum:{}, Now it is '.format(count), end='', flush=True)
                    count += 1
                except Exception as e:
                    print(f'An error occurred: {e}')
        # for spec_coll in self.SpectrumCollection_list:
        #     # merge spectra within each spectrumcollection
        #     spec_coll.merge_spectra(
        #         ms2delta=ms2delta, ppm=ppm, rt=rt, by_col_e=by_col_e)
        #     if count % 100 == 0:
        #         print('\rNum:{}, Now it is '.format(count), end='', flush=True)
        #     count += 1

    def merge_neighbor_peaks(self, threshold=1.02):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for spec_coll in self.SpectrumCollection_list:
                futures.append(executor.submit(
                    spec_coll.merge_neighbor_peaks, threshold=threshold))
            count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if count % 100 == 0:
                        print('\rNum:{}, Now it is '.format(count), end='', flush=True)
                    count += 1
                except Exception as e:
                    print(f'An error occurred: {e}')
                    raise e
        # for spec_coll in self.SpectrumCollection_list:
        #     spec_coll.merge_neighbor_peaks(threshold=threshold)

    def recal_pepmass(self):
        if len(self.SpectrumCollection_list) == 0:
            print('load file first')
            return
        for coll in self.SpectrumCollection_list:
            coll.recal_pepmass()

    def decompose_with_ions_in_mgf(self, root):
        """
        root defines the location for the decomposition results
        """
        if len(self.SpectrumCollection_list) == 0:
            print('load file first')
            return
        if not os.path.exists(root):
            os.mkdir(root)
        num = 1
        for coll in self.SpectrumCollection_list:
            coll.decompose_with_ions_in_mgf(root)
            num += 1
            if num % 100 == 0:
                print('Num:{}, Now it is '.format(
                    num) + coll.global_paras['MID'])

    def refresh_database(self, root, with_energy=True):
        """
        write the database data to new files at the defined location
        with_energy:whether saving the new database with collision energy
        """
        if len(self.SpectrumCollection_list) == 0:
            print('load file first')
            return
        if not os.path.exists(root):
            os.mkdir(root)
        num = 1
        for coll in self.SpectrumCollection_list:
            basename = os.path.basename(coll.filename)
            file_path = os.path.join(root, basename)
            with open(file_path, 'w', encoding='utf-8') as f:
                for attributes, value in coll.global_paras.items():
                    f.write('#' + str(attributes) + '=' + str(value) + '\n')
                f.write('\n')
                for spectrum in coll.spectrum_list:
                    if spectrum != None:
                        f.write(spectrum.get_mgf_string(
                            with_energy=with_energy) + "\n")
            if num % 100 == 0:
                print('Num:{}, Now it is '.format(
                    num) + coll.global_paras['MID'])
            num += 1

    def normalize_peaks(self):
        for coll in self.SpectrumCollection_list:
            coll.normalize_peaks()

    def order_peaks(self):
        for coll in self.SpectrumCollection_list:
            coll.order_peaks()

    def combine_database(self, db):
        '''
        this function combine two database into one
        does not check duplicates of spectra or comounds in the two database
        :param db: database class, to be combined
        :return: a new database
        '''

        if len(db.SpectrumCollection_list) > 0:
            combined_db = Database(foldername='', DBname=str(
                self.name) + '+' + str(db.name))
            db.SpectrumCollection_list.extend(self.SpectrumCollection_list)
            combined_db.SpectrumCollection_list = db.SpectrumCollection_list
            db.MID_ls.extend(self.MID_ls)
            combined_db.MID_ls = db.MID_ls
            db.total_ion_pairs.extend(self.total_ion_pairs)
            combined_db.total_ion_pairs = list(set(db.total_ion_pairs))
            return combined_db
        else:
            return self

    def get_frag_feature(self, minmz=30, maxmz=5000):
        """
        Summarize the fragment feature used in LDA model, all features are stored in self.frag_feature
        all different mz fragments are summarized and ordered, but not binned
        keep two decimals
        """
        print('Start summarizing frag features...')
        self.frag_feature = []
        for i, spec_c in enumerate(self.SpectrumCollection_list):
            if (i + 1) % 100 == 0:
                print('\rGetting fragment features, file num:{}'.format(
                    i + 1), end='', flush=True)
            for spec in spec_c.spectrum_list:
                peaks = spec.peaks[:, 0]
                self.frag_feature.extend(list(peaks))
        self.frag_feature = list(
            map(lambda a: round(float(a), 2), self.frag_feature))
        self.frag_feature = list(set(self.frag_feature))
        frag_feature = np.asarray(self.frag_feature)
        frag_feature = frag_feature[np.logical_and(
            frag_feature >= minmz, frag_feature <= maxmz)]
        self.frag_feature = frag_feature
        self.frag_feature.sort()
        self.frag_feature = self.frag_feature.T
        print('')
        print('frag feature calculation finished')
        print('-----------------------------------------------------------------------')

    def get_loss_feature(self, min_loss_mz=14, max_loss_mz=5000):
        """
        Summarize the neutral loss feature used in LDA model, all features are stored in self.loss_feature
        all different neutral losses are summarized and ordered, but not binned
        min_loss_mz: the minimum value for neutral loss to be kept
        """
        print('Start summarizing loss features...')
        self.loss_feature = []
        i = 0
        for spec_c in self.SpectrumCollection_list:
            i += 1
            if i % 100 == 0:
                print('\rGetting loss features, file num:{}'.format(
                    i), end='', flush=True)
            for spec in spec_c.spectrum_list:
                spec.cal_neutral_loss(min_loss_mz)
                peaks = spec.neutral_loss[:, 0]
                self.loss_feature.extend(list(peaks))
        self.loss_feature = list(
            map(lambda a: round(float(a), 2), self.loss_feature))
        # TODO make resolution a variable, e.g. 1 or 2 decimal
        self.loss_feature = list(set(self.loss_feature))
        loss_feature = np.asarray(self.loss_feature)
        loss_feature = loss_feature[np.logical_and(
            loss_feature >= min_loss_mz, loss_feature <= max_loss_mz)]
        self.loss_feature = loss_feature
        self.loss_feature.sort()
        self.loss_feature = self.loss_feature.T
        print('')
        print('loss feature calculation finished')
        print('-----------------------------------------------------------------------')

    def get_LDA_feature(self, minmz=30, maxmz=5000, min_loss_mz=14, max_loss_mz=5000):
        """
        Summarize the fragment and neutral loss feature for LDA model training, all features are stored in self.LDA_feature
        """
        if self.loss_feature is None:
            self.get_loss_feature(min_loss_mz, max_loss_mz)
        if self.frag_feature is None:
            self.get_frag_feature(minmz=minmz, maxmz=maxmz)

    def load_frag_loss_M(self, start, end, i, iontype='[M+H]+'):
        """
        Called in self.get_LDA_matrix, this method is used to construct LDA matrix (find intensity for each feature)
        in mono-processor mode
        :param start:
        :param end:
        :param i:
        :iontype
        :return: M_frag
        The result M_frag is of shape [frag_feature * spec_num]
        """
        M_frag = np.empty([len(self.frag_feature), 0], dtype=np.float)
        M_loss = np.empty([len(self.loss_feature), 0], dtype=np.float)
        num = 0
        ids = []
        for index in range(start, end):
            spec_c = self.SpectrumCollection_list[index]
            # if spec_c.spectrum_list[0].ion != iontype:
            #     continue
            for spec in spec_c.spectrum_list:
                if spec.ion != iontype:
                    continue
                c_frag = np.zeros([len(self.frag_feature), 1])
                c_loss = np.zeros([len(self.loss_feature), 1])
                pks_frag = spec.peaks
                # Keep mz to two decimal places
                if len(pks_frag) < 1:
                    continue
                pks_frag[:, 0] = np.round(pks_frag[:, 0], 2)
                # if the spectrum has no neutral loss calculated, this may happen when generate LDA feature matrix
                # based on the frag and loss features from another database
                if (not hasattr(spec, 'neutral_loss')) or (spec.neutral_loss is None):
                    spec.cal_neutral_loss()
                pks_loss = spec.neutral_loss
                if len(pks_loss) < 1:
                    continue
                pks_loss[:, 0] = np.round(pks_loss[:, 0], 2)
                for (feature, intensity) in pks_frag:
                    c_frag[np.where(self.frag_feature == feature)[
                        0]] = round(intensity, 2)
                for (feature, intensity) in pks_loss:
                    c_loss[np.where(self.loss_feature == feature)[
                        0]] = round(intensity, 2)
                M_frag = np.concatenate((M_frag, c_frag), axis=1)
                M_loss = np.concatenate((M_loss, c_loss), axis=1)
                ids.append(spec_c.MID)
                num += 1
                if num % 100 == 0:
                    print('\rprocessor {} is processing the collection {}'.format(
                        i, num), end='', flush=True)
        return M_frag, M_loss, ids

    def get_LDA_matrix(self, minmz=30, maxmz=5000, min_loss_mz=14, max_loss_mz=5000, iontype='[M+H]+'):
        '''
        generate LDA feature/intensity matrix based on given loss_feature and frag_feature (pre-generated from other database),
        if loss_feature and frag_feature are not given, generate loss_feature and frag_feature arrays based on current database
        generate lda_dict (feature name: feature value) and total_feature_matrix (size = document_num X feature num)
        :param minmz:
        :param maxmz:
        :param min_loss_mz:
        :param max_loss_mz:
        :param iontype:
        :return:
        '''
        self.get_LDA_feature(minmz=minmz, maxmz=maxmz,
                             min_loss_mz=min_loss_mz, max_loss_mz=max_loss_mz)
        # if self.loss_feature is None:
        #     self.get_loss_feature(min_loss_mz, max_loss_mz)
        # if self.frag_feature is None:
        #     self.get_frag_features(minmz, maxmz)
        print('start constructing LDA matrix...')
        st_time = time.time()
        n_cpu = int(multiprocessing.cpu_count())
        n_col = len(self.SpectrumCollection_list)
        base = int(n_col / n_cpu)
        num = n_col % n_cpu
        self.loss_feature = np.asarray(self.loss_feature)
        self.frag_feature = np.asarray(self.frag_feature)
        # test
        # self.load_frag_loss_M(0, n_col, 0, iontype=iontype)
        pool = multiprocessing.Pool(processes=n_cpu)
        Ms = []
        if n_col < n_cpu:
            for i in range(n_col):
                Ms.append(pool.apply_async(
                    self.load_frag_loss_M, (i, i + 1, i, iontype)))
        else:
            for i in range(n_cpu):
                # if i < num:
                if i + 1 < n_cpu:
                    # Ms.append(pool.apply_async(self.load_frag_loss_M, (i * base + i, (i + 1) * base + i + 1, i, iontype)))
                    Ms.append(pool.apply_async(self.load_frag_loss_M,
                              (i * base, (i + 1) * base, i, iontype)))
                else:
                    # Ms.append(pool.apply_async(self.load_frag_loss_M, (i * base + num, (i + 1) * base + num, i, iontype)))
                    Ms.append(pool.apply_async(self.load_frag_loss_M,
                              (i * base, n_col, i, iontype)))
        pool.close()
        pool.join()
        Ms_loss = []
        Ms_frag = []
        self.LDA_ids = []
        for M in Ms:
            r = M.get()
            Ms_loss.append(r[1])
            Ms_frag.append(r[0])
            self.LDA_ids.extend(r[2])
        self.LDA_ids = np.asarray(self.LDA_ids)
        # pool.terminate()
        # pool.join()
        self.loss_M = np.concatenate(Ms_loss, axis=1)
        self.frag_M = np.concatenate(Ms_frag, axis=1)
        print('')
        loss_time = time.time()
        print('LDA Matrix Construction time consuming: {}s'.format(
            loss_time - st_time))
        print('-----------------------------------------------------------------------')
        self.total_feature_M = np.concatenate(
            (self.frag_M, self.loss_M), axis=0)
        str_frag = list(map(lambda a: 'frag_' + str(a), self.frag_feature))
        str_loss = list(map(lambda a: 'loss_' + str(a), self.loss_feature))
        str_frag.extend(str_loss)
        self.lda_dict = {}
        for key, str_n in enumerate(str_frag):
            self.lda_dict[key] = str_n
        return self.total_feature_M

    def cal_LDA_feature(self, spec):
        '''
        get LDA_feauter for experimental spectra based on database-calculated feature
        '''
        pk = spec.peaks
        spec.cal_neutral_loss()
        loss = spec.neutral_loss
        pk[:, 0] = np.round(pk[:, 0], 2)
        loss[:, 0] = np.round(loss[:, 0], 2)
        c_frag = np.zeros(len(self.frag_feature))
        c_loss = np.zeros(len(self.loss_feature))
        for (feature, intensity) in pk:
            c_frag[np.where(self.frag_feature == feature)[0]] = intensity
        for (feature, intensity) in loss:
            c_loss[np.where(self.loss_feature == feature)[0]] = intensity
        return np.concatenate((c_frag, c_loss))[:, np.newaxis]

    def update_Mass(self, mass_dic):
        """
        update the database.SpectrumCollection_list[x].global_paras['Mass'] based on mass_dic dictionary
        :param mass_dic: (MID, Mass) pair dict
        :return: None
        """
        # first sort the whole database by the MID:
        self.SpectrumCollection_list.sort(key=lambda a: a.MID)
        MID_new = set(list(mass_dic.keys()))
        MID_origin = set(self.MID_ls)
        MID_common = list(MID_new.intersection(MID_origin))
        MID_common.sort()
        index = 0
        last_MID = 0
        for Spec_c in self.SpectrumCollection_list:
            if Spec_c.MID == last_MID:
                Spec_c.global_paras['Mass'] = mass_dic[MID_common[index - 1]]
            elif Spec_c.MID == MID_common[index]:
                last_MID = MID_common[index]
                Spec_c.global_paras['Mass'] = mass_dic[MID_common[index]]
                index += 1
            if index == len(MID_common):
                break

    def compare_Mass(self, mass_dic):
        '''
        check if database Mass (exact mass) is the same as the information in mass_dic
        return a list of MID with inconsistent Mass information
        '''
        if len(self.SpectrumCollection_list) == 0:
            print('load file first')
            return
        self.SpectrumCollection_list.sort(
            key=lambda a: a.MID)  # sort spectra by MID
        MID_new = set(list(mass_dic.keys()))
        MID_origin = set(self.MID_ls)
        MID_common = list(MID_new.intersection(MID_origin))
        MID_common.sort()
        index = 0
        last_MID = 0
        different_MID = []
        for Spec_c in self.SpectrumCollection_list:
            if Spec_c.MID == last_MID:
                pass
            elif Spec_c.MID == MID_common[index]:
                last_MID = MID_common[index]
                if abs(float(Spec_c.global_paras['Mass']) - mass_dic[MID_common[index]]) > 0.01:
                    different_MID.append(Spec_c.MID)
                index += 1
            if index == len(MID_common):
                break
        return different_MID

    def sort_by_charges(self, charge=1):
        '''
        only keep SpectraumCollection with required charge
        '''
        new_ls = []
        for spec_c in self.SpectrumCollection_list:
            if abs(spec_c.charge) == charge:
                new_ls.append(spec_c)
        self.SpectrumCollection_list = new_ls

    def get_cartain_ion_spectrum(self, quest):
        """
        filter self.SpectrumCollection according to the quest
        :param quest: ('+ or -', ion) e.g. ('+','Na')
        :return:
        """
        new_ls = []
        for spec_c in self.SpectrumCollection_list:
            if spec_c.ion_pair in quest:
                new_ls.append(spec_c)
        self.SpectrumCollection_list = new_ls

    # def remove_odd_spec(self):
    #     new_ls = []
    #     for spec_c in self.SpectrumCollection_list:
    #         # 如果feature最小值小于0.01直接删除
    #         if np.min(np.abs(spec_c.spectrum_list[0].all_feature)) < 0.01:
    #             continue
    #         else:
    #             spec_c.spectrum_list[0].remove_negative_neutral_loss()
    #             new_ls.append(spec_c)
    #     self.SpectrumCollection_list = new_ls


class SpectrumCollection:
    def __init__(self, filename, type='ms2'):
        self.filename = filename
        self.spectrum_list = []
        self.scandict = {}
        self.type = type
        # ms1list contains the mz and rt pair for all precusor ions in the SpectrumCollection
        self.ms1list = pd.DataFrame(columns=['mz', 'rt'])

    def load_from_file(self, drop_mslevels=[1], peaktable=False, ppm=20, rt=30,
                       inten_thresh=0, engine='UTF-8', remove_pre=False, min_mz_num=1):
        '''
        :remove_pre: remove peaks +-17 precursor ion
        :min_mz_num: minimum numbers of peaks in a spectrum
        :inten_thresh: after normalization, the peaks with intensity below inten_thresh will be removed
        '''
        # distinguish the format of the file
        extension = os.path.splitext(self.filename)[1]
        try:
            if extension.lower() == ".mzxml":
                self.load_from_mzXML(drop_mslevels=drop_mslevels, peaktable=peaktable, ppm=ppm, rt=rt,
                                    min_mz_num=min_mz_num, engine=engine, inten_thresh=inten_thresh)
            elif extension.lower() == ".mgf":
                self.load_from_mgf(min_mz_num=min_mz_num,
                                engine=engine, inten_thresh=inten_thresh)
            else:
                print('only mzxml and mgf files are supported')
        except Exception as e:
            print("error: ",self.filename,"  ",e)
            
        # filter based on inten_threshold (calculate absolute intensity threshold based on relative inten_threshold),
        # then check min_mz_num,then remove precursor peaks, then check min_mz_num
        # for spec in self.spectrum_list:
        #    spec.normalize_peaks()

        if inten_thresh > 0:
            i = 0
            new_ls = []
            while i < len(self.spectrum_list):
                spec = self.spectrum_list[i]
                if type(spec.peaks) is not np.ndarray:
                    spec.peaks = np.asarray(spec.peaks, dtype=np.float)
                real_thresh = inten_thresh / 100.0 * max(spec.peaks[:, 1])
                index = np.where(spec.peaks[:, 1] > real_thresh)[0]
                if len(index) >= min_mz_num and len(index) > 0:
                    spec.peaks = spec.peaks[index]
                    new_ls.append(spec)
                i += 1
            self.spectrum_list = new_ls

        if remove_pre:
            i = 0
            new_ls = []
            while i < len(self.spectrum_list):
                spec = self.spectrum_list[i]
                index = np.where(np.abs(spec.peaks[:, 0] - spec.mz) > 17)[0]
                if len(index) >= min_mz_num and len(index) > 0:
                    spec.peaks = spec.peaks[index]
                    new_ls.append(spec)
                i += 1
            self.spectrum_list = new_ls

        # for spec in self.spectrum_list:
        #     spec.normalize_peaks()

        # if a spectrum is removed from spectrum_list, the ms1list needs to be changed accordingly
        new_ls = self.spectrum_list
        new_scandict = {}
        # new_ms1list = pd.DataFrame(columns={'mz','rt'})
        new_ms1list = [pd.DataFrame()]
        for spectrum in new_ls:
            new_scandict[spectrum.scan] = spectrum
            new_ms1list.append(pd.DataFrame.from_records(
                [{'mz': spectrum.mz, 'rt': spectrum.retention_time}]))
            # new_ms1list = new_ms1list.append({'mz':spectrum.mz,'rt':spectrum.retention_time}, ignore_index=True)
        new_ms1list = pd.concat(new_ms1list, ignore_index=True)
        self.scandict = new_scandict
        self.ms1list = new_ms1list

    def load_from_mgf(self, inten_thresh=0, engine='UTF-8', min_mz_num=6):
        '''
        load SpectrumCollection from mgf file
        '''
        charge = 0
        mz = 0
        peaks = []
        scan = 0
        spectrum_count = 0
        non_empty_spectrum = 0
        self.global_paras = {}
        ion = None
        ion_pair = None
        energy = 0
        inchikey = ''
        # Formula = ''
        # Name = ''
        # Mass = None

        output_spectra = []
        self.ion_list = []
        # print(self.filename)
        try:
            f = open(self.filename, "r", encoding=engine)
            lines = f.readlines()
        except:
            try:
                f = open(self.filename, "r", encoding='GBK')
                lines = f.readlines()
            except:
                try:
                    f = open(self.filename, "r", encoding='GB2312')
                    lines = f.readlines()
                except:
                    print('please give the correct encoding')
        for line in lines:
            if line.startswith('\ufeff'):
                line = line[1:]
            mgf_file_line = line.rstrip()
            if len(mgf_file_line) < 4:
                continue

            if mgf_file_line[0] == "#":
                # which means annotation
                r = mgf_file_line[1:]
                try:
                    attribute, value = r.split('=')
                    self.global_paras[attribute] = value
                    if attribute.lower() == 'inchikey':
                        self.global_paras["inchikey"] = value
                except:
                    continue
                continue

            if mgf_file_line == "BEGIN IONS":
                # initialize
                charge = 0
                mz = 0
                peaks = []
                scan += 1
                ion = None
                ion_pair = None
                energy = -1
                inchikey = ''
                formula = ''
                mid = ''
                mass = -1
                continue
            if mgf_file_line == "END IONS":
                # if spectrum_count % 10000 == 0:
                #     print("Spectra Loaded\t%d\tReal\t%d" % (spectrum_count, non_empty_spectrum))

                if len(peaks) > 0:
                    non_empty_spectrum += 1
                    # normalize th spectrum
                    if inchikey == '' and 'inchikey' in self.global_paras:
                        inchikey = self.global_paras['inchikey']
                    adding_spectrum = Spectrum(self.filename, scan, -1, peaks, mz, charge, 2, inchikey=inchikey,
                                               ion=ion, mid=mid, ion_pair=ion_pair, collision_energy=energy, mass=mass, formula=formula)
                    # adding_spectrum.normalize_peaks()
                    if inten_thresh is not None:
                        if type(adding_spectrum.peaks) is not np.ndarray:
                            adding_spectrum.peaks = np.asarray(
                                adding_spectrum.peaks, dtype=np.float)
                        real_thresh = inten_thresh / 100.0 * \
                            max(adding_spectrum.peaks[:, 1])
                        adding_spectrum.filter_by_intensity(
                            inten_thresh=real_thresh)
                    if len(adding_spectrum.peaks) >= min_mz_num:
                        output_spectra.append(adding_spectrum)
                spectrum_count += 1
                continue
            if mgf_file_line.split('=')[0] == 'TITLE':
                try:
                    ion = mgf_file_line.split(' ')[1]
                    energy = float(mgf_file_line.split(' ')
                                   [0].split('=')[1][:-1])
                    # flag is a char
                    ion_info = ion.split('M')[1].split(']')[
                        0]  # ion_info indicates addition
                    flag = ion.split('M')[0].split(
                        '[')[1]  # flag indicate how many M
                    if len(flag) < 1:
                        flag = '1'
                    # [M+H2O-2H]2- ion_pair is ('1','+H2O-2H')
                    ion_pair = (flag, ion_info)
                    if ion_pair not in self.ion_list:
                        self.ion_list.append(ion_pair)
                except:
                    continue
            if mgf_file_line.split('=')[0] == 'ION':
                ion = mgf_file_line.split('=')[1]
                
                if ion.lower() != "none":
                    if '[' in ion.lower():
                        ion_info = ion.split('M')[1].split(']')[
                            0]  # ion_info indicates addition
                        flag = ion.split('M')[0].split(
                            '[')[1]  # flag indicate how many M
                    else:
                        ion_info = ion.split('M')[1]
                        flag = ion.split('M')[0] # flag indicate how many M
                    if len(flag) < 1:
                        flag = '1'
                    # [M+H2O-2H]2- ion_pair is ('1','+H2O-2H')
                    ion_pair = (flag, ion_info)
                    if ion_pair not in self.ion_list:
                        self.ion_list.append(ion_pair)
                continue
            if mgf_file_line.split('=')[0] == 'COLLISION_ENERGY':
                try:
                    energy = float(mgf_file_line.split('=')[1])
                except:
                    energy = -1
                continue
            if 'inchikey' in mgf_file_line.split('=')[0].lower():
                try:
                    inchikey = str(mgf_file_line.split('=')[1])
                except:
                    inchikey = ""
                continue
            if mgf_file_line[:8] == "PEPMASS=":
                mz = float(mgf_file_line[8:])
                continue
            if mgf_file_line[:7] == "CHARGE=":
                try:
                    if mgf_file_line[7:].find("-") != -1:
                        charge = - int(mgf_file_line[7:].replace("-", ""))
                    else:
                        charge = int(mgf_file_line[7:].replace("+", ""))
                except:
                    charge = 0

                continue
            if mgf_file_line[:4] == "SEQ=":
                peptide = mgf_file_line[4:]
                continue
            if mgf_file_line[:8] == "PROTEIN=":
                protein = mgf_file_line[8:]
                continue
            if mgf_file_line.startswith("MASS="):
                try:
                    mass = float(mgf_file_line.split('=')[1])
                except:
                    mass = -1
                continue
            if mgf_file_line.startswith("FORMULA="):
                try:
                    formula = str(mgf_file_line.split('=')[1])
                except:
                    formula = ""
                continue
            if mgf_file_line.startswith("MID="):
                try:
                    mid = str(mgf_file_line.split('=')[1])
                except:
                    mid = ""
                continue
            if mgf_file_line.find("=") == -1:
                # save the m/z and the intensity
                peak_split = re.split("[ |\t]+", mgf_file_line)
                peaks.append([float(peak_split[0]), float(peak_split[1])])
        self.spectrum_list = output_spectra
        self.MID = str(os.path.basename(
            self.filename).split('.')[0][0:].split('_')[0])
        self.global_paras[
            'MID'] = self.MID  # in some mgf files, the MID has no suffix, make sure MID is based on filename suffix

    def load_from_mzXML(self, drop_mslevels=[1], peaktable=False, ppm=20, rt=30, min_mz_num=1, engine='GBK',
                        inten_thresh=0):
        '''
        :drop_mslevels: a list containing mslevels that need to be removed
        :ppm: use to check if match with entries in peaktable
        :rt: use to check if match with entries in peaktable
        '''

        def read_mzxml_scan(scan, index, struct_iter_ok, canary, drop_mslevels):
            ms_level = int(scan['@msLevel'])

            if ms_level in drop_mslevels:
                return ms_level, None, struct_iter_ok, canary

            scan_number = int(scan['@num'])
            collision_energy = 0.0
            fragmentation_method = "NO_FRAG"

            try:
                collision_energy = float(scan['@collisionEnergy'])
            except KeyboardInterrupt:
                raise
            except:
                collision_energy = -1

            # Optional fields
            base_peak_intensity = 0.0
            base_peak_mz = 0.0
            base_peak_intensity = float(scan.get('@basePeakIntensity', 0.0))
            base_peak_mz = float(scan.get('@basePeakMz', 0.0))
            totIonCurrent = 0

            try:
                totIonCurrent = float(scan.get('@totIonCurrent', 0.0))
            except KeyboardInterrupt:
                raise
            except:
                fragmentation_method = "NO_FRAG"

            try:
                precursor_mz_tag = scan['precursorMz']
                precursor_mz = float(precursor_mz_tag['#text'])
                precursor_scan = int(
                    precursor_mz_tag.get('@precursorScanNum', 0))
                precursor_charge = int(
                    precursor_mz_tag.get('@precursorCharge', 0))
                precursor_intensity = float(
                    precursor_mz_tag.get('@precursorIntensity', 0))

                try:
                    fragmentation_method = precursor_mz_tag['@activationMethod']
                except KeyboardInterrupt:
                    raise
                except:
                    fragmentation_method = "NO_FRAG"

            except KeyboardInterrupt:
                raise
            except:
                if ms_level == 2:
                    raise

            # Loading retention time
            retention_time = 0.0
            try:
                retention_time_string = scan['@retentionTime']
                # print(retention_time_string)
                # 以秒为单位
                retention_time = float(retention_time_string[2:-1])
            except KeyboardInterrupt:
                raise
            except:
                print("ERROR")
                retention_time = 0.0

            peaks_precision = float(scan['peaks'].get('@precision', '32'))
            peaks_compression = scan['peaks'].get('@compressionType', 'none')
            peak_string = scan['peaks'].get('#text', '')
            if canary and peak_string != '':
                try:
                    peaks = decode_spectrum(
                        peak_string, peaks_precision, peaks_compression, struct_iter_ok)
                except:
                    struct_iter_ok = False
                canary = False
            if peak_string != '':
                try:
                    peaks = decode_spectrum(
                        peak_string, peaks_precision, peaks_compression, struct_iter_ok)
                except:
                    peaks = []
                canary = False
            else:
                peaks = []
            if ms_level == 1:
                if len(peaks) == 0:
                    output = None
                else:
                    if len(peaks) >= min_mz_num:
                        output = Spectrum(self.filename,
                                          scan_number,
                                          index,
                                          peaks,
                                          mz=0,
                                          charge=0,
                                          ms_level=ms_level,
                                          rt=retention_time,
                                          collision_energy=collision_energy,
                                          fragmentation_method=fragmentation_method,
                                          precursor_intensity=precursor_intensity,
                                          totIonCurrent=totIonCurrent
                                          )
                        # output.normalize_peaks()
                        if inten_thresh is not None:
                            if type(output.peaks) is not np.ndarray:
                                output.peaks = np.asarray(
                                    output.peaks, dtype=np.float)
                            real_thresh = inten_thresh / \
                                100.0 * max(output.peaks[:, 1])
                            output.filter_by_intensity(
                                inten_thresh=real_thresh)
                        if len(output.peaks) < min_mz_num:
                            output = None
                    else:
                        output = None
            if ms_level == 2:
                if len(peaks) < min_mz_num:
                    output = None
                else:
                    output = Spectrum(self.filename,
                                      scan_number,
                                      index,
                                      peaks,
                                      precursor_mz,
                                      precursor_charge,
                                      ms_level,
                                      rt=retention_time,
                                      collision_energy=collision_energy,
                                      fragmentation_method=fragmentation_method,
                                      precursor_intensity=precursor_intensity,
                                      totIonCurrent=totIonCurrent
                                      )

                    # output.normalize_peaks()

                    if inten_thresh is not None:
                        if type(output.peaks) is not np.ndarray:
                            output.peaks = np.asarray(
                                output.peaks, dtype=np.float)
                        real_thresh = inten_thresh / \
                            100.0 * max(output.peaks[:, 1])
                        output.filter_by_intensity(inten_thresh=real_thresh)

                    if len(output.peaks) < min_mz_num:
                        output = None
                    # output.retention_time = retention_time
            return ms_level, output, struct_iter_ok, canary

        def load_mzxml_file(filename, drop_mslevels=[1], peaktable=False, ppm=20, rt=30, engine=engine):
            output_ms2 = []

            struct_iter_ok = True
            canary = True

            with open(filename, encoding=engine) as fd:
                xmltodict_start = time.time()
                mzxml = xmltodict.parse(fd.read())
                xmltodict_end = time.time()
                print("XML time: " + str(xmltodict_end - xmltodict_start))
                # extract all scans
                read_scans = mzxml['mzXML']['msRun']['scan']
                index = 1
                for i, scan in enumerate(read_scans):
                    match = False
                    if scan.get('@msLevel', -1) == '2':
                        if peaktable is not False:
                            precursormass = round(
                                float(scan['precursorMz']['#text']), 4)
                            retention = round(
                                float(scan['@retentionTime'][2:-1]), 2)
                            mzlist = np.array(peaktable['mz'].astype('float'))
                            mzlist = list(
                                abs((mzlist - precursormass) / precursormass))
                            rtlist = np.array(peaktable['rt'].astype('float'))
                            rtlist = list(abs(rtlist - retention))
                            for j in range(len(mzlist)):
                                if mzlist[j] < (ppm / 1e06) and rtlist[j] < rt:
                                    match = True
                                    break
                        if ((peaktable is not False) and (match)) or (peaktable is False):
                            ms_level, spectrum, struct_iter_ok, canary = read_mzxml_scan(scan, index, struct_iter_ok,
                                                                                         canary, drop_mslevels)
                            if ms_level == 2 and spectrum is not None:
                                output_ms2.append(spectrum)
                                index += 1
                    print('\r{}, {} scans loaded, progressing: {}%'.format(os.path.basename(filename), i + 1,
                                                                           round((i + 1) / len(read_scans) * 100, 2)),
                          end='', flush=True)
            print('')
            return output_ms2

        self.spectrum_list = load_mzxml_file(self.filename, drop_mslevels=drop_mslevels, peaktable=peaktable, ppm=ppm,
                                             rt=rt)
        file_idx = os.path.split(self.filename)[1]
        # Do indexing on scan number
        num = 0
        new_ms1list = [self.ms1list]
        for spectrum in self.spectrum_list:
            self.scandict[spectrum.scan] = spectrum
            self.scandict[file_idx + ":" + str(spectrum.scan)] = spectrum
            # self.ms1list = pd.concat([self.ms1list,pd.DataFrame.from_records([{'mz':spectrum.mz,'rt':spectrum.retention_time}])],ignore_index=True)
            # self.ms1list = self.ms1list.append({'mz':spectrum.mz,'rt':spectrum.retention_time}, ignore_index=True)
            new_ms1list.append(pd.DataFrame.from_records(
                [{'mz': spectrum.mz, 'rt': spectrum.retention_time}]))
            num += 1
            print('\r{}, {} mslist constructed, progressing: {}%'.format(os.path.basename(self.filename), num,
                                                                         round(num / len(self.spectrum_list) * 100, 2)),
                  end='', flush=True)
        print('')
        print('-----------------------------------------------------------------------')
        self.ms1list = pd.concat(new_ms1list, ignore_index=True)
        self.MID = 'None'  # if load from .mzxml file, MID is assigned as 'None'

    # TODO: remove this load_ms1_mzxml_file() function, combine it with load_from_mzXML() function
    def load_ms1_mzxml_file(self, scan_span=None, rt_span=None, mz_span=None, if_norm=False, tol=0.01, inten_thresh=2):
        '''
        read mzxml files only containing mslevel 1 data, and merge all scans according to scan_span/rt_span/mz_span
        mz will be rounded to two decimals
        :scan_span: e.g. [1,10]
        :if_norm: if True, each scan will be normalized to the max intensity in that scan, after merge normalize again
        :tol: if mz difference between two peaks below tol, they will be merged, intensity be added together
        '''

        def read_scan_ms1(scan, index):
            ms_level = int(scan['@msLevel'])
            fragmentation_method = "NO_FRAG"
            assert ms_level == 1
            scan_number = int(scan['@num'])
            try:
                collision_energy = float(scan['@collisionEnergy'])
            except KeyboardInterrupt:
                raise
            except:
                collision_energy = -1
            totIonCurrent = 0

            try:
                totIonCurrent = float(scan.get('@totIonCurrent', 0.0))
            except KeyboardInterrupt:
                raise
            except:
                fragmentation_method = "NO_FRAG"
            try:
                retention_time_string = scan['@retentionTime']
                retention_time = float(retention_time_string[2:-1])
            except KeyboardInterrupt:
                raise
            except:
                print("ERROR")
                retention_time = 0.0
            peaks_precision = float(scan['peaks'].get('@precision', '32'))
            peaks_compression = scan['peaks'].get('@compressionType', 'none')
            peak_string = scan['peaks'].get('#text', '')
            # TODO: 不要使用这个值进行filter，保留每一个scan，mz_span的部分
            # mz = float(scan['@basePeakMz'])
            if peak_string != '':
                peaks = decode_spectrum(
                    peak_string, peaks_precision, peaks_compression, True)
            else:
                peaks = []
                # if len(peaks) == 0:
                #     output = None
                # else:
                #     output = Spectrum(self.filename,
                #                       scan_number,
                #                       index,
                #                       peaks,
                #                       mz=0,
                #                       charge=0,
                #                       ms_level
                #                       )
                #     return output
            if len(peaks) == 0:
                output = None
            else:
                output = Spectrum(self.filename,
                                  scan_number,
                                  index,
                                  peaks,
                                  mz=0,
                                  charge=0,
                                  ms_level=ms_level,
                                  rt=retention_time,
                                  collision_energy=collision_energy,
                                  fragmentation_method=fragmentation_method,
                                  precursor_intensity=None,
                                  totIonCurrent=totIonCurrent
                                  )
            return output

        output_ms1 = []
        with open(self.filename, encoding='utf-8') as fd:
            xmltodict_start = time.time()
            mzxml = xmltodict.parse(fd.read())
            xmltodict_end = time.time()
            print("XML time: " + str(xmltodict_end - xmltodict_start))
            read_scans = mzxml['mzXML']['msRun']['scan']  # extract all scans
            filename_output = os.path.split(self.filename)[1]
            index = 1
            for scan in read_scans:
                if scan.get('@msLevel', -1) == '1':
                    spec = read_scan_ms1(scan, index)
                    index += 1
                    if spec is not None:
                        if inten_thresh is not None:
                            temp = spec.peaks
                            # filter signals below inten_thresh
                            spec.peaks = temp[temp[:, 1] > inten_thresh, :]
                        output_ms1.append(spec)
        up_scan = None
        low_scan = None
        up_rt = None
        low_rt = None
        up_mz = None
        low_mz = None
        if scan_span is not None:
            assert hasattr(scan_span, '__iter__')
            assert len(scan_span) == 2
            up_scan = scan_span[1]
            low_scan = scan_span[0]
        if rt_span is not None:
            assert hasattr(rt_span, '__iter__')
            assert len(rt_span) == 2
            up_rt = rt_span[1]
            low_rt = rt_span[0]
        if mz_span is not None:
            assert hasattr(mz_span, '__iter__')
            assert len(mz_span) == 2
            up_mz = mz_span[1]
            low_mz = mz_span[0]
        r = []
        all_spec = np.zeros((0, 2))
        for spec in output_ms1:
            if up_scan is not None and (spec.scan > up_scan or spec.scan < low_scan):
                continue
            if up_rt is not None and (spec.retention_time > up_rt or spec.retention_time < low_rt):
                continue
            if up_mz is not None:
                index_l = np.where(spec.peaks[:, 0] >= low_mz)[0]
                index_u = np.where(spec.peaks[:, 0] <= up_mz)[0]
                indexes = list(set(index_u) & set(index_l))
                spec.peaks = spec.peaks[indexes]
            if if_norm:
                spec.normalize_peaks()
            all_spec = np.concatenate((all_spec, spec.peaks), axis=0)
            r.append(spec)
        all_spec = all_spec[np.argsort(all_spec[:, 0])]  # sort according to mz
        # r.sort(key=lambda a: a.mz)
        merged_spec = np.zeros((0, 2))
        if len(r) > 0:
            peak = all_spec[0, 0]
            inten = all_spec[0, 1]
            for i in range(1, all_spec.shape[0]):
                if (all_spec[i, 0] - peak) < tol:
                    # peak = (peak + all_spec[i, 0]) / 2 #peaks are too close, if use average almost all peaks are merged
                    inten = inten + all_spec[i, 1]
                else:
                    merged_spec = np.concatenate(
                        (merged_spec, np.asarray([[round(peak, 4), inten]])), axis=0)
                    peak = all_spec[i, 0]
                    inten = all_spec[i, 1]
            merged_spec = np.concatenate(
                (merged_spec, np.asarray([[peak, inten]])), axis=0)
            if if_norm:
                merged_spec[:, 1] = merged_spec[:, 1] / \
                    (max(merged_spec[:, 1]))
            return merged_spec
        else:
            return None

    def search_spectrum(self, otherspectrum, pm_tolerance, peak_tolerance, min_matched_peaks, min_score,
                        analog_search=False, top_k=1):
        '''
        find cosine similar spectra from a SpectrumCollection based on a given spectrum
        :return: a list of dictionary, match_obj["filename"]/match_obj["scan"]/match_obj["queryfilename"]
        match_obj["queryscan"]/match_obj["cosine"]/match_obj["matchedpeaks"]/match_obj["mzerror"]
        '''
        if otherspectrum == None:
            return []

        if len(otherspectrum.peaks) < min_matched_peaks:
            return []

        match_list = []
        for myspectrum in self.spectrum_list:
            if myspectrum == None:
                continue

            if len(myspectrum.peaks) < min_matched_peaks:
                continue

            mz_delta = abs(myspectrum.mz - otherspectrum.mz)
            if mz_delta < pm_tolerance or analog_search == True:
                cosine_score, matched_peaks = myspectrum.cosine_spectrum(
                    otherspectrum, peak_tolerance)
                # Also check for min matched peaks
                if cosine_score > min_score and matched_peaks >= min_matched_peaks:
                    # print(cosine_score, matched_peaks, mz_delta)
                    match_obj = {}
                    match_obj["filename"] = myspectrum.filename
                    match_obj["scan"] = myspectrum.scan
                    match_obj["queryfilename"] = otherspectrum.filename
                    match_obj["queryscan"] = otherspectrum.scan
                    match_obj["cosine"] = cosine_score
                    match_obj["matchedpeaks"] = matched_peaks
                    match_obj["mzerror"] = abs(
                        myspectrum.mz - otherspectrum.mz)

                    match_list.append(match_obj)

        match_list = sorted(
            match_list, key=lambda score_obj: score_obj["cosine"])

        return match_list[:min(len(match_list), top_k)]

    # updates both the scans and the index
    def make_scans_sequential(self):
        self.scandict = {}
        scan = 1
        for spectrum in self.spectrum_list:
            spectrum.scan = scan
            spectrum.index = scan - 1
            self.scandict[scan] = spectrum
            scan += 1

    def make_ms1list_sequential(self):
        '''
        sort spectra according to mz and rt
        for mslevel=1 spectra, mz=0
        '''
        self.ms1list.reset_index(drop=True, inplace=True)
        if len(self.ms1list) > 0:
            self.ms1list = self.ms1list.sort_values(by=['mz', 'rt'])
            self.spectrum_list = list(
                map(self.spectrum_list.__getitem__, self.ms1list.index))
            self.ms1list.reset_index(drop=True, inplace=True)

    def merge_neighbor_peaks(self, threshold=1.02):
        def find_nth_largest(arr, N):
            if N >= len(arr):
                return None,None
            sorted_indices = arr[:, 1].argsort()[::-1]

            nth_largest_index = sorted_indices[N]
            nth_largest_value = arr[nth_largest_index]
            return nth_largest_value, nth_largest_index
        
        for i_spec in range(len(self.spectrum_list)):
            arr = self.spectrum_list[i_spec].peaks
            # Sort the array by mass in ascending order
            arr = arr[arr[:, 0].argsort()]
            # Find the element with the highest inten
            # max_item = arr[arr[:, 1].argmax()]
            done_count = 0
            max_item,max_index =  find_nth_largest(arr,done_count)
            while max_item is not None:
                to_remove = np.zeros(len(arr), dtype=bool)
                # max_index = np.where(np.all(arr == max_item, axis=1))[0][0]
                left_index = max_index - 1

                while left_index >= 0:
                    if abs(arr[left_index][0] - max_item[0]) <= threshold:
                        to_remove[left_index] = True
                        left_index -= 1
                    else:
                        left_index = -1

                right_index = max_index + 1

                while right_index < len(arr):
                    if abs(arr[right_index][0] - max_item[0]) <= threshold:
                        to_remove[right_index] =True
                        right_index += 1
                    else:
                        right_index = 1000000

                # for i in sorted(remove_indices, reverse=True):
                #     arr = np.delete(arr, i, axis=0)
                arr = arr[~to_remove]
                done_count+=1
                max_item,max_index =  find_nth_largest(arr,done_count)
                if max_index == None:
                    break
                # done_list = np.append(done_list, max_item[0])
                # mask = np.isin(arr[:, 0], done_list, invert=True)
                # # apply the mask to the array to get the filtered array
                # filtered_arr = arr[mask, :]
                # if len(filtered_arr) == 0:
                #     break
                # # get the index of the maximum item based on the 'inten' key
                # max_index = np.argmax(filtered_arr[:, 1])
                # # Find the next highest inten element that hasn't been processed yet
                # max_item = filtered_arr[max_index]
            self.spectrum_list[i_spec].peaks = arr
            
            
    def merge_neighbor_peaks_(self, threshold=1.02):
        for i_spec in range(len(self.spectrum_list)):
            # Sort the array by mass in ascending order
            peaks = self.spectrum_list[i_spec].peaks
            peaks = peaks[np.argsort(peaks[:, 0])]

            # Boolean array to mark peaks for removal
            to_remove = np.zeros(len(peaks), dtype=bool)

            # Iterate over the peaks once, marking neighbors for removal
            i = 0
            while i < len(peaks):
                # Find the range of indices where peaks are within the threshold
                close_by = np.abs(peaks[i+1:, 0] - peaks[i, 0]) <= threshold
                if np.any(close_by):
                    # Mark the neighboring peaks for removal, keeping the one with the highest intensity
                    neighbor_indices = np.where(close_by)[0] + i + 1
                    max_intensity_index = neighbor_indices[np.argmax(peaks[neighbor_indices, 1])]
                    to_remove[neighbor_indices] = True
                    to_remove[max_intensity_index] = False  # Keep the peak with the highest intensity
                    i = max_intensity_index + 1  # Skip ahead to the next unmarked peak
                else:
                    i += 1
            # Remove the marked peaks
            self.spectrum_list[i_spec].peaks = peaks[~to_remove]
    def merge_spectra(self, ms2delta=0.01, ppm=20, rt=30, by_col_e=True):
        extension = os.path.splitext(self.filename)[1]
        if extension.lower() == '.mzxml':
            self.merge_mzxml_spectra(
                ppm=ppm, rt=rt, ms2delta=ms2delta, by_col_e=by_col_e)
        elif extension.lower() == '.mgf':
            self.merge_mgf_spectra(ms2delta=ms2delta)
        else:
            print('only mzxml and mgf file types are supported')
        # self.normalize_peaks()

    def merge_mgf_spectra(self, ms2delta=0.01):
        '''
        for mgf files, require one file only contains spectra from the same precursor.
        merge_mgf_spectra combine all spectra in the same spectrumcollection (from the same file)
        '''
        num_spectra = len(self.spectrum_list)
        # if there is only one spectrum in th collection then do nothing
        if num_spectra < 2:
            return
        all_spec = self.spectrum_list[0].peaks
        i = 1
        while i < num_spectra:
            all_spec = np.concatenate(
                (all_spec, self.spectrum_list[i].peaks), axis=0)
            # merge peaks of the two spectra, peaks is a list of tuple, each tuple contains mz and intensity
            i += 1
        all_spec = all_spec[np.argsort(all_spec[:, 0])]
        merged_spec = np.zeros((0, 2))
        peak = all_spec[0, 0]
        inten = all_spec[0, 1]
        for i in range(1, all_spec.shape[0]):
            if (all_spec[i, 0] - peak) < ms2delta:
                # peak = (peak + all_spec[i, 0]) / 2
                inten = inten + all_spec[i, 1]
            else:
                merged_spec = np.concatenate(
                    (merged_spec, np.asarray([[round(peak, 4), inten]])), axis=0)
                peak = all_spec[i, 0]
                inten = all_spec[i, 1]
        merged_spec = np.concatenate(
            (merged_spec, np.asarray([[peak, inten]])), axis=0)
        self.spectrum_list = [self.spectrum_list[0]]
        self.spectrum_list[0].peaks = merged_spec
        # after merge, collision_energy has been combined
        self.spectrum_list[0].collision_energy = -1

    def merge_mzxml_spectra(self, ppm=20, rt=30, ms2delta=0.01, by_col_e=True):
        """
        ppm: the tolerance, the indicator to determine whether the two mz are the same
        rt : retention time tolerance, with the acceptance of ppm, bearing which means they are two identical things
        """
        print('-----------------------------------------------------------------------')
        print('Start merging spectra' +
              ' by Collision Energy' if by_col_e else '')
        # make the m/z in order first
        if not by_col_e:
            self.make_ms1list_sequential()
            num_spectra = len(self.spectrum_list)
            if num_spectra < 2:
                return
            i = 0
            process = 0
            total_process = num_spectra
            while i < (num_spectra - 1):
                mz1 = self.ms1list.iloc[i]['mz']
                mz2 = self.ms1list.iloc[i + 1]['mz']
                rt1 = self.ms1list.iloc[i]['rt']
                rt2 = self.ms1list.iloc[i + 1]['rt']
                if (abs((mz1 - mz2) / mz2 * 1e6) < ppm) and (abs(rt1 - rt2) < rt):
                    # merge rows in ms1list
                    mz_new = round((mz1 + mz2) / 2, 4)
                    retention_new = round((rt1 + rt2) / 2, 2)
                    rowindex = self.ms1list.index[i + 1]
                    self.ms1list.drop(rowindex, inplace=True)
                    self.ms1list.iloc[i]['mz'] = mz_new
                    self.ms1list.iloc[i]['rt'] = retention_new
                    # merge peaks of the two spectra, peaklist is a list of tuple, each tuple contains mz and intensity
                    peaklist1 = self.spectrum_list[i].peaks
                    peaklist2 = self.spectrum_list[i + 1].peaks
                    self.spectrum_list[i].peaks = merge_peaklist(
                        peaklist1, peaklist2, ms2delta)
                    self.spectrum_list[i].filename = self.spectrum_list[i].filename + '+' + self.spectrum_list[
                        i + 1].filename
                    # after merge, collision_enrgy and scan values will be discarded
                    self.spectrum_list[i].collision_energy = -1
                    self.spectrum_list[i].scan = -1
                    self.spectrum_list[i].mz = mz_new
                    self.spectrum_list[i].retention_time = retention_new
                    del self.spectrum_list[i + 1]
                    num_spectra = num_spectra - 1
                    process += 1
                else:
                    i = i + 1
                    process = process + 1
                print('\r {} spectrums merged, Merging progressing: {}%'.format(process,
                                                                                round(process / total_process * 100,
                                                                                      2)), end='', flush=True)

            new_scandict = {}
            # new_ms1list = pd.DataFrame(columns={'mz','rt'})
            new_ms1list = [pd.DataFrame()]
            new_ls = self.spectrum_list
            for spectrum in new_ls:
                new_scandict[spectrum.scan] = spectrum
                # new_ms1list = pd.concat([new_ms1list,pd.DataFrame.from_records([{'mz':spectrum.mz,'rt':spectrum.retention_time}])],ignore_index=True)
                # new_ms1list = new_ms1list.append({'mz':spectrum.mz,'rt':spectrum.retention_time}, ignore_index=True)
                new_ms1list.append(pd.DataFrame.from_records(
                    [{'mz': spectrum.mz, 'rt': spectrum.retention_time}]))
            self.scandict = new_scandict
            self.ms1list = pd.concat(new_ms1list, ignore_index=True)
            print('')
        else:
            # 首先把spectrum list中的谱图按照collision energy分类
            # self.spec_0 = []
            # self.spec_10 = []
            # self.spec_20 = []
            # self.spec_40 = []
            # self.spec_60 = []
            # self.spec_80 = []
            # for spec in self.spectrum_list:
            #     if spec.collision_energy == 0:
            #         self.spec_0.append(spec)
            #     if spec.collision_energy == 10:
            #         self.spec_10.append(spec)
            #     if spec.collision_energy == 20:
            #         self.spec_20.append(spec)
            #     if spec.collision_energy == 40:
            #         self.spec_40.append(spec)
            #     if spec.collision_energy == 60:
            #         self.spec_60.append(spec)
            #     if spec.collision_energy == 80:
            #         self.spec_80.append(spec)

            col_e_ls = [a.collision_energy for a in self.spectrum_list]
            col_e_ls = list(set(col_e_ls))
            col_e_ls.sort()
            spec_l_col = [[] for _ in range(
                len(col_e_ls))]  # empty list of lists for spectrum lists corresponding to different collision energy

            for spec in self.spectrum_list:
                spec_l_col[col_e_ls.index(spec.collision_energy)].append(spec)
            # spec_l_col corresponds to collision energy in col_e_ls

            self.spectrum_list = []

            # mul_ls = [0, 10, 20, 40, 60, 80]
            for type, spec_l in enumerate(spec_l_col):
                # 将每一个collision energy的集合按照mz和retention time从大到小的顺序排列
                spec_l.sort(key=lambda a: a.retention_time)
                spec_l.sort(key=lambda a: a.mz)
                num_spectra = len(spec_l)
                total_process = num_spectra
                process = 0
                if num_spectra >= 2:
                    i = 0
                    while i < (num_spectra - 1):
                        mz1 = spec_l[i].mz
                        mz2 = spec_l[i + 1].mz
                        rt1 = spec_l[i].retention_time
                        rt2 = spec_l[i + 1].retention_time
                        if (abs((mz1 - mz2) / mz2 * 1e6) < ppm) and (abs(rt1 - rt2) < rt):
                            # merge rows in ms1list
                            mz_new = round((mz1 + mz2) / 2, 4)
                            retention_new = round((rt1 + rt2) / 2, 2)
                            # merge peaks of the two spectra, peaklist is a list of tuple, each tuple contains mz and intensity
                            peaklist1 = spec_l[i].peaks
                            peaklist2 = spec_l[i + 1].peaks
                            spec_l[i].peaks = merge_peaklist(
                                peaklist1, peaklist2, ms2delta)
                            spec_l[i].mz = mz_new
                            spec_l[i].retention_time = retention_new
                            # after merge scan number will be discarded
                            spec_l[i].scan = -1
                            del spec_l[i + 1]
                            num_spectra = num_spectra - 1
                            process += 1
                        else:
                            i = i + 1
                            process += 1
                        print('\rCollision Energy {}, {} spectrums merged, Merging progressing: {}%'.format(
                            col_e_ls[type], process + 1, round((process + 1) / (total_process) * 100, 2)), end='',
                            flush=True)
                    print('')
                self.spectrum_list.extend(spec_l)

                new_scandict = {}
                # new_ms1list = pd.DataFrame(columns={'mz','rt'})
                new_ms1list = [pd.DataFrame()]
                new_ls = self.spectrum_list
                for spectrum in new_ls:
                    new_scandict[spectrum.scan] = spectrum
                    # new_ms1list = pd.concat([new_ms1list,pd.DataFrame.from_records([{'mz':spectrum.mz,'rt':spectrum.retention_time}])],ignore_index=True)
                    # new_ms1list = new_ms1list.append({'mz':spectrum.mz,'rt':spectrum.retention_time}, ignore_index=True)
                    new_ms1list.append(pd.DataFrame.from_records(
                        [{'mz': spectrum.mz, 'rt': spectrum.retention_time}]))
                self.scandict = new_scandict
                self.ms1list = pd.concat(new_ms1list, ignore_index=True)

        print('Merging spectrum finished')
        print('-----------------------------------------------------------------------')

    def normalize_peaks(self):
        for spec in self.spectrum_list:
            spec.normalize_peaks()

    def order_peaks(self):
        for spec in self.spectrum_list:
            spec.order_peaks()

    def save_to_mgf(self, output_mgf, renumber_scans=True):
        '''
        write data to mgf file one spectrum after another
        :output_mgf: open file link
        '''
        if renumber_scans == True:
            self.make_scans_sequential()
        for spectrum in self.spectrum_list:
            if spectrum != None:
                output_mgf.write(spectrum.get_mgf_string() + "\n")

    def save_to_tsv(self, output_tsv_file, mgf_filename="", renumber_scans=True):
        if renumber_scans == True:
            self.make_scans_sequential()
        output_tsv_file.write(self.spectrum_list[0].get_tsv_header() + "\n")
        for spectrum in self.spectrum_list:
            if spectrum != None:
                output_tsv_file.write(
                    spectrum.get_tsv_line(mgf_filename) + "\n")

    def save_to_sptxt(self, output_sptxt_file):
        for spectrum in self.spectrum_list:
            if spectrum != None:
                output_sptxt_file.write(spectrum.get_sptxt_string() + "\n")

    def decompose_with_ions_in_mgf(self, root):
        """
        This function decompose the mgf file with different ions, the scans of the identical ions will be written to a same file named after the ions
        this function can write database results into new .mgf files
        """
        if len(self.ion_list) == 1:
            if not os.path.exists(root):
                os.mkdir(root)
            ion_pair = self.ion_list[0]
            basename = self.global_paras['MID'] + '_' + str(ion_pair) + '.mgf'
            file_path = os.path.join(root, basename)
            with open(file_path, 'w', encoding='utf-8') as f:
                for attributes, value in self.global_paras.items():
                    try:
                        f.write('#' + attributes + '=' + value + '\n')
                    except:
                        print(file_path, '#' + attributes + '=' + value)
                f.write('\n')
                for spectrum in self.spectrum_list:
                    if spectrum != None:
                        f.write(spectrum.get_mgf_string() + "\n")
        else:
            if not os.path.exists(root):
                os.mkdir(root)
            for ion_pair in self.ion_list:
                basename = self.global_paras['MID'] + \
                    '_' + str(ion_pair) + '.mgf'
                file_path = os.path.join(root, basename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    for attributes, value in self.global_paras.items():
                        try:
                            f.write('#' + attributes + '=' + value + '\n')
                        except:
                            print(file_path, '#' + attributes + '=' + value)
                    f.write('\n')
                    for spec in self.spectrum_list:
                        if spec.ion_pair == ion_pair:
                            f.write(spec.get_mgf_string() + '\n')

    def recal_pepmass(self):
        '''
        TODO [M+H2O-H] how to calculate
        calculate spectrum.mz based on Mass and iontype, e.g. for [M+Na]+, mz=Mass+23
        '''
        # save the pep mass under the mz attribute
        for spec in self.spectrum_list:
            if spec.ion is not None:
                count = float(spec.ion_pair[0])  # [2M+H-H2O]+, count is 2
                adducts = spec.ion_pair[1]  # [2M+H-H2O]+, adducts is '+H-H2O'
                try:
                    adducts = re.split('([\+\-]{1})', adducts)
                    # '+H-H2O', adducts is ['+','H','-','H2O']
                    adducts = [i for i in adducts if i != '']
                    mz = count * float(self.global_paras['Mass'])
                    for i in range(0, len(adducts), 2):
                        flag = adducts[i]  # '+'/'-'
                        ele = adducts[i+1]
                        ele = re.split('(^\d+)', ele)
                        ele = [i for i in ele if i != '']  # e.g. ['2','H2O']
                        if flag == '+':
                            if len(ele) > 1:
                                num = int(ele[0])
                                ele = str(ele[1])
                            else:
                                num = 1
                                ele = str(ele[0])
                            mz = mz + num * element_mass_dict[ele]
                        elif flag == '-':
                            if len(ele) > 1:
                                num = int(ele[0])
                                ele = str(ele[1])
                            else:
                                num = 1
                                ele = str(ele[0])
                            mz = mz - num * element_mass_dict[ele]
                        else:
                            spec.mz = float(-1)
                            return
                    # divide by charge state
                    spec.mz = round(mz / abs(spec.charge), 4)
                except:
                    spec.mz = float(-1)

    def get_all_peaklists(self):
        r = []
        for spec in self.spectrum_list:
            r.append(spec.peaks)
        return r

    def get_certain_rt_time(self, lower_b, upper_b):
        '''
        :return: SpectrumCollection only containing spectra within the retention time range
        '''
        new_ls = []
        new_scandict = {}
        # new_ms1list = pd.DataFrame(columns={'mz','rt'})
        new_ms1list = [pd.DataFrame()]
        for spec in self.spectrum_list:
            if spec.retention_time >= lower_b and spec.retention_time <= upper_b:
                new_ls.append(spec)
        for spectrum in new_ls:
            new_scandict[spectrum.scan] = spectrum
            # self.scandict[file_idx + ":" + str(spectrum.scan)] = spectrum
            # new_ms1list = pd.concat(
            #     [new_ms1list, pd.DataFrame.from_records([{'mz': spectrum.mz, 'rt': spectrum.retention_time}])],
            #     ignore_index=True)
            # new_ms1list = new_ms1list.append({'mz':spectrum.mz,'rt':spectrum.retention_time}, ignore_index=True)
            new_ms1list.append(pd.DataFrame.from_records(
                [{'mz': spectrum.mz, 'rt': spectrum.retention_time}]))
        new_SC = SpectrumCollection(self.filename)
        new_SC.scandict = new_scandict
        new_SC.ms1list = pd.concat(new_ms1list, ignore_index=True)
        new_SC.spectrum_list = new_ls
        return new_SC

    def get_certain_mz(self, mz, thresh_hold=0.01):
        '''
        :thresh_hold: mz within the +- thresh-hold will be considered as the same precursor ion
        :return: SpectrumCollection only containing spectra from the given precursor ion mz
        '''
        new_ls = []
        new_scandict = {}
        # new_ms1list = pd.DataFrame(columns={'mz','rt'})
        new_ms1list = [pd.DataFrame()]
        for spec in self.spectrum_list:
            if abs(spec.mz - mz) <= thresh_hold:
                new_ls.append(spec)
        for spectrum in new_ls:
            new_scandict[spectrum.scan] = spectrum
            # self.scandict[file_idx + ":" + str(spectrum.scan)] = spectrum
            # new_ms1list = pd.concat(
            #     [new_ms1list, pd.DataFrame.from_records([{'mz': spectrum.mz, 'rt': spectrum.retention_time}])],
            #     ignore_index=True)
            # new_ms1list = new_ms1list.append({'mz':spectrum.mz,'rt':spectrum.retention_time}, ignore_index=True)
            new_ms1list.append(pd.DataFrame.from_records(
                [{'mz': spectrum.mz, 'rt': spectrum.retention_time}]))
        new_SC = SpectrumCollection(self.filename)
        new_SC.scandict = new_scandict
        new_SC.ms1list = pd.concat(new_ms1list, ignore_index=True)
        new_SC.spectrum_list = new_ls
        return new_SC

    # def cal_LDA_features(self, frag_feature, loss_feature):
    #     assert len(self.spectrum_list) == 1, 'This is not a merged Database, Please merge it first'
    #     pks = self.spectrum_list[0].peaks
    #     pks[:, 0] = np.round(pks[:, 0], 2)
    #     c = np.zeros(len(frag_feature))
    #     for (feature, intensity) in pks:
    #         c[np.where(frag_feature == feature)[0]] = intensity
    #     neutral_loss = self.spectrum_list[0].cal_neutral_loss()
    #     c_ = np.zeros(len(loss_feature))
    #     for (feature, intensity) in neutral_loss:
    #         c_[np.where(loss_feature == feature)[0]] = intensity
    #     total = np.append(c, c_)
    #     return total[np.newaxis]


class Spectrum:
    def __init__(self, filename, scan, index, peaks, mz, charge, ms_level, rt=0, ion=None, ion_pair=None, mass="", mid="", formula="", inchikey="",
                 collision_energy=-1, fragmentation_method="NO_FRAG", precursor_intensity=0.0, totIonCurrent=0.0):
        """
        when initializing the Spectrum Class instance, the initialization
        process hte raw peaks and normalize the peaks intensity into 100 scale
        """
        self.filename = filename
        self.scan = scan
        # make sure that "peaks" is in ndarray type
        self.peaks = peaks if type(peaks) is np.ndarray else np.asarray(peaks)

        # check if "peaks" ar normalized
        self.mz = mz

        self.charge = charge
        self.index = index
        self.ms_level = ms_level
        # ion is a string
        self.ion = ion
        self.ion_pair = ion_pair
        self.retention_time = rt
        self.collision_energy = collision_energy
        self.fragmenation_method = fragmentation_method
        self.precursor_intensity = precursor_intensity
        self.totIonCurrent = totIonCurrent
        self.neutral_loss = None
        self.mass = mass
        self.mid = mid
        self.inchikey = inchikey
        self.formula = formula
        # self.all_features = np.append(self.peaks[:, 0], self.neutral_loss[:, 0])

    def get_mgf_string(self, with_energy=True):
        output_lines = []
        output_lines.append("BEGIN IONS")
        output_lines.append("MID=" + str(self.mid))
        output_lines.append("SCANS=" + str(self.scan))
        output_lines.append("PEPMASS=" + str(self.mz))
        output_lines.append("INCHIKEY=" + str(self.inchikey))
        output_lines.append("ION=" + str(self.ion))
        output_lines.append("MASS=" + str(self.mass))
        output_lines.append("FORMULA=" + str(self.formula))
        # make sure the charge attribute always has a sign
        output_lines.append("CHARGE=" + str(self.charge) if self.charge <=
                            0 else "CHARGE=" + '+' + str(self.charge))
        if with_energy:
            output_lines.append("COLLISION_ENERGY=" +
                                str(self.collision_energy))
        output_lines.append(self.get_mgf_peak_string())
        output_lines.append("END IONS")

        return "\n".join(output_lines)

    def get_mgf_peak_string(self):
        output_string = ""
        for peak in self.peaks:
            output_string += str(peak[0]) + "\t" + str(peak[1]) + "\n"

        return output_string

    @staticmethod
    def get_tsv_header():
        return "filename\tspectrumindex\tspectrumscan\tcharge\tmz"

    def get_max_mass(self):
        max_mass = 0.0
        for peak in self.peaks:
            max_mass = max(max_mass, peak[0])
        return max_mass

    # Straight up cosine between two spectra
    def cosine_spectrum(self, other_spectrum, peak_tolerance):
        total_score, reported_alignments = spectrum_alignment.score_alignment(self.peaks, other_spectrum.peaks,
                                                                              self.mz * self.charge,
                                                                              other_spectrum.mz * other_spectrum.charge,
                                                                              peak_tolerance, self.charge)
        return total_score, len(reported_alignments)

    # Looks at windows of a given size, and picks the top peaks in there
    # TODO:add a method to filter by intensity
    def filter_by_intensity(self, inten_thresh=2):
        if type(self.peaks) is not np.ndarray:
            self.peaks = np.asarray(self.peaks, dtype=np.float)
        self.peaks = self.peaks[np.argwhere(
            self.peaks[:, 1] > inten_thresh)[:, 0]]

    def window_filter_peaks(self, window_size, top_peaks):
        new_peaks = window_filter_peaks(self.peaks, window_size, top_peaks)
        self.peaks = new_peaks

    def filter_to_top_peaks(self, top_k_peaks):
        sorted_peaks = sorted(
            self.peaks, key=lambda peak: peak[1], reverse=True)
        sorted_peaks = sorted_peaks[:top_k_peaks]
        sorted_peaks = sorted(
            sorted_peaks, key=lambda peak: peak[0], reverse=False)
        self.peaks = sorted_peaks

    def filter_precursor_peaks(self):
        new_peaks = filter_precursor_peaks(self.peaks, 17, self.mz)
        self.peaks = new_peaks

    def filter_noise_peaks(self, min_snr):
        # noise * snr latter quater
        average_noise_level = numerical_utilities.calculate_noise_level_in_peaks(
            self.peaks)
        new_peaks = []
        for peak in self.peaks:
            if peak[1] > average_noise_level * min_snr:
                new_peaks.append(peak)
        self.peaks = new_peaks

    def filter_peak_mass_range(self, lower, higher):
        new_peaks = []
        for peak in self.peaks:
            if peak[0] < lower or peak[0] > higher:
                new_peaks.append(peak)
        self.peaks = new_peaks

    def generated_spectrum_vector(self, peptide=None, attenuation_ratio=0.0, tolerance=0.5, bin_size=1):
        peaks_to_vectorize = self.peaks
        max_mass = 1500

        if peptide != None:
            charge_set = range(1, self.charge + 1)
            theoretical_peaks = psm_library.create_theoretical_peak_map(self.peptide,
                                                                             ["b", "y", "b-H2O", "b-NH3", "y-H2O",
                                                                              "y-NH3", "a"], charge_set=charge_set)
            annotated_peaks, unannotated_peaks = psm_library.extract_annotated_peaks(theoretical_peaks, self.peaks,
                                                                                          tolerance)
            new_peaks = annotated_peaks
            if attenuation_ratio > 0:
                for unannotated_peak in unannotated_peaks:
                    unannotated_peak[1] *= attenuation_ratio
                    new_peaks.append(unannotated_peak)
            peaks_to_vectorize = sorted(new_peaks, key=lambda peak: peak[0])
        # Doing
        peak_vector = numerical_utilities.vectorize_peaks(
            peaks_to_vectorize, max_mass, bin_size)
        return peak_vector

    def get_number_of_signal_peaks(self, SNR_Threshold=5):
        return numerical_utilities.calculate_signal_peaks_in_peaklist(self.peaks, SNR_Threshold)

    def get_number_of_peaks_within_percent_of_max(self, percent=1.0):
        max_peak_intensity = 0.0
        for peak in self.peaks:
            max_peak_intensity = max(peak[1], max_peak_intensity)
        intensity_threshold = percent / 100.0 * max_peak_intensity
        number_of_peaks = 0
        for peak in self.peaks:
            if peak[1] > intensity_threshold:
                number_of_peaks += 1
        return number_of_peaks

    """Gives sum of intensity of all spectrum peaks"""

    def get_total_spectrum_intensity(self):
        total_peak_intensity = 0
        for peak in self.peaks:
            total_peak_intensity += peak[1]
        return total_peak_intensity

    def normalize_peaks(self):
        self.peaks[:, 1] /= (np.max(self.peaks[:, 1]) / 100)

    def order_peaks(self):
        self.peaks[:, 1] = np.sort(self.peaks[:, 1])

    def cal_neutral_loss(self, min_loss_mz=14, max_loss_mz=5000):
        self.neutral_loss = np.stack(
            (self.mz - self.peaks[:, 0], self.peaks[:, 1])).T
        self.neutral_loss = self.neutral_loss[np.where(
            self.neutral_loss[:, 0] >= min_loss_mz)[0]]
        self.neutral_loss = self.neutral_loss[np.where(
            self.neutral_loss[:, 0] <= max_loss_mz)[0]]
        return self.neutral_loss

    def remove_negative_neutral_loss(self):
        self.neutral_loss = self.neutral_loss[np.where(
            self.neutral_loss[:, 0] < 0)[0]]


#

if __name__ == '__main__':
    control_dir = ''
    sample_dir = ''
    result_dir = ''

    # mz_span=()
    # rt_span=()

    # mzXML_file = '../Ants mix-MSMS.mzXML'

    # query_collection = SpectrumCollection(mzXML_file)
    # pk = query_collection.load_ms1_mzxml_file(rt_span=rt_span, mz_span=mz_span, if_norm=True)
    # pks = [pk1, pk2, pk3, pk4, pk5]
    # 样本来源（文件名称） mz 丰度
    # TODO: 存下矩阵/图片
    # m = plot_merge(pks, plot_engine='matplotlib')
    # database_pth = '../Database_merged_'
    # db = Database(database_pth, 'tb')
    # db.load_files(inten_thresh=2, engine='utf-8', n_workers=16)
    # db.get_frag_features()
    # db.get_loss_feature()
    # M = db.get_LDA_matrix()
    # lda = LDA(n_components=500, n_jobs=-1, learning_method='online', batch_size=256)
    # st_t = time.time()
    # lda.fit(M.T)
    # end_t = time.time()
    # print('The whle training time is {}'.format(end_t - st_t))

    #     neutral loss: figure out the reason
    #     TODO:input spectrum, user-define-threshold output: all the ids whose cos score larger than the threshold
    #     refine the component_matrix after LDA training
    #     gips sampling converge
    #
    # original_db = Database('../Database', 'or_d')
    # original_db.load_files()
    # result_ls = []
    # for spec_c in original_db.SpectrumCollection_list:
    #     if float(spec_c.global_paras['Mass']) == 0:
    #         result_ls.append(spec_c.filename)
    # from Library.utils import export_ls2txt
    # export_ls2txt(result_ls, 'Mass_error.txt')

    # db = Database('../Database_merged_', 'db')
    # db.load_files()
    # db.get_loss_feature()
    # r_ls =[]
    # for spec_c in db.SpectrumCollection_list:
    #     spec = spec_c.spectrum_list[0]
    #     r_ls.append([spec_c.MID, spec.ion, np.min(np.abs(spec.neutral_loss[:, 0]))])
    # r_ls.sort(key=lambda a:a[2], reverse=True)
    # from Library.utils import export_ls2txt
    # export_ls2txt(r_ls, 'loss_peak的绝对值最小值.txt')

    # db = Database('../Database_merged_', 'db')
    # db.load_files(inten_thresh=2)
    # db.get_loss_feature()
    # from Library.utils import export_odd_neutral_loss
    # export_odd_neutral_loss(db)

    # db.filter_noise_peaks(5)
    # result_ls = []
    # for spec_c in db.SpectrumCollection_list:
    #     if len(spec_c.spectrum_list[0].peaks) == 0:
    #         result_ls.append(spec_c.filename)

    # db = Database('../Database_merged', 'dd')
    # db.load_files()
    # f = pd.read_excel('../metlinMS1.xlsx',engine='openpyxl')
    # pairs = f.values[:, :2]
    # mass_dic = {}
    # for (MID,Mass) in pairs:
    #     mass_dic[int(MID)] = Mass
    # db.update_Mass(mass_dic)
    # db.recal_pepmass()
    # db.refresh_database('../Database_merged_', False)

    # f = pd.read_excel('../metlinMS1.xlsx',engine='openpyxl')
    # pairs = f.values[:, :2]
    # mass_dic = {}
    # for (MID,Mass) in pairs:
    #     mass_dic[int(MID)] = Mass
    # r_ls = db.compare_Mass(mass_dic)

    # [7, 8, 9, 14, 18, 21, 24, 25, 35, 52, 21053, 21638, 23633]
    db = Database('/Users/Peter1/Desktop/testDB', 'testDB')
    db.load_files(min_mz_num=0, remove_pre=False, engine='utf-8')
    db.decompose_with_ions_in_mgf('Database_ions')
# TODO(seems done already)：更改计算merge_spectrum的方式，先全部合并之后，然后merge
# TODO: 先去掉precursor，norm，calculate——neutral——loss， 400以上的frag_feature去掉， difference_between_fragment, 差值小于18去掉
