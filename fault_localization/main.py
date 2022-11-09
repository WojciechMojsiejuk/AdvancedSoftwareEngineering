import csv
import glob
import pandas as pd
from pandas.errors import ParserError
from scipy.stats import rankdata
import warnings
from tqdm import tqdm
import numpy as np
from faults_extractor import faults_extractor
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


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
            self.exam = None
            self.prerank = None
            self.calc_jaccard()
            self.calc_dstar()
            self.calc_dstar(3)
            self.calc_dstar(5)
            self.calc_tarantula()
            self.calc_ochiai()
            self.calc_naish()
            self.calc_gp08()
            self.calc_gp10()
            self.calc_gp10()
            self.calc_gp11()
            self.calc_gp13()
            self.calc_gp20()
            self.calc_gp26()
            self.prerank = self.coverage

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

    def calc_tarantula(self):
        tarantula_top = self.coverage['Ef']/(self.coverage['Ef'] + self.coverage['Nf'])
        tarantula_bottom = self.coverage['Ep']/(self.coverage['Ep'] + self.coverage['Np'])
        self.coverage['Tarantula'] = tarantula_top / (tarantula_top + tarantula_bottom)
        self.formulas.append('Tarantula')

    def calc_ochiai(self):
        ochiai_bottom = np.sqrt((self.coverage['Ef'] + self.coverage['Nf'])*(self.coverage['Ef'] + self.coverage['Ep']))
        self.coverage['Ochiai'] = self.coverage['Ef'] / ochiai_bottom
        self.formulas.append('Ochiai')

    def calc_naish(self):
        self.coverage['Naish'] = self.coverage['Ef'] - (self.coverage['Ep']/ (self.coverage['Ep'] + self.coverage['Np'] + 1))
        self.formulas.append('Naish')

    def calc_gp08(self):
        self.coverage['GP08'] = self.coverage['Ef']**(2*self.coverage['Ep']+2*self.coverage['Ef']+3*self.coverage['Np'])
        self.formulas.append('GP08')

    def calc_gp10(self):
        self.coverage['GP10'] = np.sqrt(np.abs(self.coverage['Ef'] - 1/self.coverage['Np']))
        self.formulas.append('GP10')

    def calc_gp11(self):
        self.coverage['GP11'] = self.coverage['Ef']**2*(self.coverage['Ef']**2+np.sqrt(self.coverage['Np']))
        self.formulas.append('GP11')

    def calc_gp13(self):
        self.coverage['GP13'] = self.coverage['Ef'] * (1 + 1/(2*self.coverage['Ep']+self.coverage['Ef']))
        self.formulas.append('GP13')

    def calc_gp20(self):
        self.coverage['GP20'] = 2*(self.coverage['Ef'] + self.coverage['Np']/(self.coverage['Ep']+self.coverage['Np']))
        self.formulas.append('GP20')

    def calc_gp26(self):
        self.coverage['GP26'] = 2*self.coverage['Ef'] ** 2 + np.sqrt(self.coverage['Np'])
        self.formulas.append('GP26')

    def calc_rank(self, formula):
        self.coverage['Rank'] = rankdata(self.coverage[formula], method='min')  # rank data based on coefficient
        self.coverage = self.coverage.sort_values('Rank', ascending=False)  # sort data by rank

    def revert(self):
        self.coverage = self.prerank

    def calc_top(self, top):
        top_values = self.coverage['Rank'].unique()[:top] # find top N unique values in rank
        self.coverage_top = self.coverage.loc[self.coverage['Rank'].isin(top_values)]  # select top N unique values in dataframe

    def check_within_top(self):
        return np.sum(self.coverage_top['Line'].isin(self.faults['Line'])) == self.number_of_faults_covered  # if we found all faults

    def calc_exam(self, normalized=False):
        self.exam = np.max(np.where(self.coverage['Line'].isin(self.faults['Line'])))
        if normalized:
            self.exam = self.exam / self.coverage.shape[0]



data_path = './eval-data/eval/'
TOP = 20
if __name__ == '__main__':
    coverage_paths = glob.glob(data_path+'*.coverage.csv')  # find all coverage files
    faulty_paths = glob.glob(data_path+'*.faulty-lines.csv')  # find all faulty-lines files

    data_size = len(coverage_paths)
    count_all = data_size
    failed = 0
    exams_all = []
    number_of_formulas = 0
    for idx in tqdm(range(data_size)):
        fault = FaultLocalizator(coverage_paths[idx], faulty_paths[idx])
        if idx == 0:
            number_of_formulas = len(fault.formulas)
            count = np.zeros(number_of_formulas)  # counter how many times faulty line was found within TOP N search
            exams = np.zeros(number_of_formulas)
        if not fault.FAILED_STATUS:
            for i, formula in enumerate(fault.formulas):
                fault.calc_rank(formula)
                fault.calc_exam(True)
                exams[i] = fault.exam
                fault.calc_top(TOP)
                if fault.check_within_top():
                    count[i] += 1
                fault.revert()
            exams_all.append(exams.copy())
        else:
            failed += 1
            count_all -= 1
    exams_all = np.array(exams_all)
    count /= count_all

    plt.figure()
    plt.boxplot(exams_all, labels=fault.formulas)
    plt.xticks(rotation=90)
    plt.ylabel('Normalized Exam score')
    plt.xlabel('Formulas')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(count, '.')
    plt.ylabel('Accuracy')
    plt.xlabel('Formulas')
    plt.xticks(range(len(count)), fault.formulas, rotation=90)
    title_str = f'Top {TOP}'
    plt.title(title_str)
    plt.tight_layout()
    plt.show()