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
from sklearn.model_selection import train_test_split
matplotlib.use('TkAgg')


class NotFoundInTheDataFrame(Exception):
    pass


class FaultLocalizator:

    def __init__(self, coverage_filepath, faults_filepath=None, benchmark=False):
        try:
            self.faults = None
            self.coverage = pd.read_csv(coverage_filepath, skiprows=2, engine='python', sep=',')  # load coverage file, skip first two rows
            if faults_filepath is not None:
                self.faults = faults_extractor(faults_filepath)
                self.number_of_faults_covered = np.sum(self.coverage['Line'].isin(self.faults['Line']))
                if self.number_of_faults_covered == 0:
                    raise NotFoundInTheDataFrame()
            self.FAILED_STATUS = False
            self.formulas = []
            self.exam = None
            self.prerank = None
            if benchmark:
                self.calc_jaccard()
            else:
                self.calc_dstar()
                self.calc_dstar(3)
                # self.calc_dstar(5)
                self.calc_tarantula()
                self.calc_ochiai()
                self.calc_naish()
                # self.calc_gp08()
                # self.calc_gp10()
                # self.calc_gp11()
                # self.calc_gp13()
                # self.calc_gp20()
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
        self.coverage['GP08'] = self.coverage['Ef']**2*(self.coverage['Ep']+2*self.coverage['Ef']+3*self.coverage['Np']) # TODO: check this formula
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
        top_values = set(self.coverage['Rank'][:top]) # find what ranks are within N values in rank
        self.coverage_top = self.coverage.loc[self.coverage['Rank'].isin(top_values)]  # select top N unique values in dataframe

    def check_within_top(self):
        if self.faults is not None:
            return np.sum(self.coverage_top['Line'].isin(self.faults['Line'])) == self.number_of_faults_covered  # if we found all faults
        else:
            return self.coverage['Line'].isin(self.coverage_top['Line']).astype(int)

    def calc_exam(self, normalized=False):
        self.exam = np.max(np.where(self.coverage['Line'].isin(self.faults['Line'])))
        if normalized:
            self.exam = self.exam / self.coverage.shape[0]

    def calc_majority(self, z_score):
        self.coverage['Majority'] = z_score


data_path = './eval-closure-issue-fixed/'
TOP = 20
TOP_SEARCH = [5, 10, 25, 50, 100, 200, 300, 400, 500, 750,  1000]
SEED = 1
PLOTS = True


if __name__ == '__main__':
    coverage_paths = glob.glob(data_path+'*.coverage.csv')  # find all coverage files
    faulty_paths = glob.glob(data_path+'*.faulty-lines.csv')  # find all faulty-lines files

    coverage_train, coverage_test = train_test_split(coverage_paths, test_size=0.2, random_state=SEED)
    faulty_train, faulty_test = train_test_split(faulty_paths, test_size=0.2, random_state=SEED)

    data_size = len(coverage_train)
    count_all = data_size
    failed = 0
    exams_all = []
    number_of_formulas = 0
    benchmark_exam = None
    benchmark_exam_all = []
    # Training

    for idx in tqdm(range(data_size)):
        fault = FaultLocalizator(coverage_train[idx], faulty_train[idx], benchmark=False)
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

        benchmark = FaultLocalizator(coverage_train[idx], faulty_train[idx], benchmark=True)

        if idx == 0:
            number_of_formulas = len(benchmark.formulas)
            b_count = np.zeros(number_of_formulas)  # counter how many times faulty line was found within TOP N search
            b_exams = np.zeros(number_of_formulas)
        if not fault.FAILED_STATUS:
            for i, formula in enumerate(benchmark.formulas):
                benchmark.calc_rank(formula)
                benchmark.calc_exam(True)
                benchmark_exam = benchmark.exam
                benchmark.calc_top(TOP)
                if benchmark.check_within_top():
                    b_count[i] += 1
                benchmark.revert()
            benchmark_exam_all.append(benchmark_exam.copy())
        else:
            failed += 1
            count_all -= 1

    exams_all = np.array(exams_all)
    training_exams_copy = exams_all
    count /= count_all

    if PLOTS:
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

    # prior information of the performance of the formulas on training
    # z = rankdata(count, 'dense')
    # z = np.exp(count)
    # z = np.exp(count) * np.square(rankdata(count, 'dense'))
    # z = np.exp((count - np.min(count))/(np.max(count) - np.min(count)))
    z = (count - np.min(count)) / (np.max(count) - np.min(count)) + 1
    z_all = np.sum(z)





    data_size = len(coverage_test)
    count_all = data_size
    failed = 0
    number_of_formulas = 0
    exams_all = []

    # Testing

    for idx in tqdm(range(data_size)):
        fault = FaultLocalizator(coverage_test[idx])
        if idx == 0:
            number_of_formulas = len(fault.formulas)

        if not fault.FAILED_STATUS:
            count = np.zeros((fault.coverage.shape[0],
                              number_of_formulas))  # counter how many times faulty line was found within TOP N search
            for i, formula in enumerate(fault.formulas):
                if formula == 'Jaccard':
                    continue
                for top in TOP_SEARCH:
                    fault.calc_rank(formula)
                    fault.calc_top(top)
                    fault.revert() # return to pre-sorted array
                    faults_within_top = fault.check_within_top()
                    count[:, i] += faults_within_top
            z_score = (count @ z) / z_all
            fault.calc_majority(z_score)
            fault.calc_rank('Majority')
            fault.faults = faults_extractor(faulty_test[idx])
            fault.number_of_faults_covered = np.sum(fault.coverage['Line'].isin(fault.faults['Line']))
            if fault.number_of_faults_covered != 0:
                fault.calc_exam(True)
                exams_all.append(fault.exam)
            else:
                warnings.warn(f'Fault with file {faulty_test[idx]}')
                warnings.warn('Line from faulty csv was not found in the whole dataframe')
        else:
            failed += 1
            count_all -= 1


    plt.figure()
    plt.subplot(1,2,1)
    plt.boxplot(exams_all)
    plt.xticks(rotation=90)
    plt.ylabel('Normalized Exam score')
    plt.xlabel('Formulas')
    plt.title('Our model')
    plt.tight_layout()


    plt.subplot(1,2,2)
    plt.boxplot(benchmark_exam_all)
    plt.xticks(rotation=90)
    plt.ylabel('Normalized Exam score')
    plt.xlabel('Formulas')
    plt.title('Jaccard')
    plt.tight_layout()
    plt.show()

    print(np.sum(count))
    print(count.shape)
    print(z)
    print(z_all)

    # Calculates the Wilcoxon signed rank test

    forFormula = 'Jaccard'
    forFormulaIndex = fault.formulas.index(forFormula)
    forFormulaExam = training_exams_copy[forFormulaIndex]

    p_scores = []
    p_score_formulas = []
    for i, formula in enumerate(fault.formulas):
        if formula == forFormula:
            continue

        compareToExam = training_exams_copy[i]
        p_scores.append(stats.wilcoxon(forFormulaExam, compareToExam)[1])
        p_score_formulas.append(formula)

    plt.scatter(p_score_formulas, p_scores)
    plt.plot([0, len(p_score_formulas) - 2], [0.05, 0.05], 'k-', color='red')
    plt.ylabel('p scores')
    plt.yscale('log')
    plt.xlabel('Formulas')
    plt.title(forFormula + ' Wilcoxon signed ranked test')
    plt.show()

