import csv
import glob
import pandas as pd
from pandas.errors import ParserError
from scipy.stats import rankdata
import warnings
from tqdm import tqdm
import numpy as np
from faults_extractor import faults_extractor


class NotFoundInTheDataFrame(Exception):
    pass


class FaultLocalizator:

    def __init__(self, coverage_filepath, faults_filepath):
        try:
            self.coverage = pd.read_csv(coverage_filepath, skiprows=2, engine='python', sep=',')  # load coverage file, skip first two rows
            self.faults = faults_extractor(faults_filepath)
            self.number_of_faults_covered = np.sum(self.coverage['Line'].isin(self.faults['Line']))
            if self.number_of_faults_covered == 0:
                raise NotFoundInTheDataFrame()
            self.FAILED_STATUS = False
            self.formulas = []
        except ParserError as pe:
            warnings.warn(f'Fault with file {coverage_paths[idx]}')
            warnings.warn(pe)
            self.FAILED_STATUS = True
        except NotFoundInTheDataFrame:
            warnings.warn(f'Fault with file {coverage_paths[idx]}')
            warnings.warn('Line from faulty csv was not found in the whole dataframe')
            self.FAILED_STATUS = True

    def calc_jaccard(self):
        self.coverage['Jaccard'] = self.coverage['Ef'] / (self.coverage['Ef'] + self.coverage['Nf'] + self.coverage['Np'])  # calculate Jaccard similarity coefficient
        self.formulas.append('Jaccard')

    def calc_dstar(self, d=2):
        dstar_name = 'DStart'+str(d)
        self.coverage[dstar_name] = self.coverage['Ef'] ** d / (self.coverage['Ep'] + self.coverage['Nf'])
        self.formulas.append(dstar_name)

    def calc_rank(self, formula):
        self.coverage['Rank'] = rankdata(self.coverage[formula], method='min')  # rank data based on coefficient
        self.coverage.sort_values('Rank', ascending=False)  # sort data by rank

    def calc_top(self, top):
        top_values = self.coverage['Rank'].unique()[:top] # find top N unique values in rank
        self.coverage_top = self.coverage.loc[self.coverage['Rank'].isin(top_values)]  # select top N unique values in dataframe

    def check_within_top(self):
        return np.sum(self.coverage_top['Line'].isin(self.faults['Line'])) == self.number_of_faults_covered  # if we found all faults

data_path = './eval-data/eval/'
TOP = 5
if __name__ == '__main__':
    coverage_paths = glob.glob(data_path+'*.coverage.csv')  # find all coverage files
    faulty_paths = glob.glob(data_path+'*.faulty-lines.csv')  # find all faulty-lines files

    data_size = len(coverage_paths)
    count_all = data_size
    count = 0  # counter how many times faulty line was found within TOP N search
    failed = 0
    for idx in tqdm(range(data_size)):

        fault = FaultLocalizator(coverage_paths[idx], faulty_paths[idx])
        if not fault.FAILED_STATUS:
            fault.calc_jaccard()
            fault.calc_rank(fault.formulas[0])
            fault.calc_top(TOP)
            if fault.check_within_top():
                count += 1
        else:
            failed += 1
            count_all -= 1

    acc = count/count_all
    print(f'Accuracy to find faulty line within top {TOP} = {acc} with {failed} out of {data_size} failed files')
