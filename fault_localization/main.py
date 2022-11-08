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

        try:
            df = pd.read_csv(coverage_paths[idx], skiprows=2, engine='python', sep=',')  # load coverage file, skip first two rows
        except ParserError as pe:
            warnings.warn(f'Fault with file {coverage_paths[idx]}')
            warnings.warn(pe)
            failed += 1
            count_all -= 1  # we must decrease a number of all element as this one cannot be used for evaluation
            continue

        df['Jaccard'] = df['Ef'] / (df['Ef'] + df['Nf'] + df['Np'])  # calculate Jaccard similarity coefficient
        df['DStar2'] = df['Ef']**2 / (df['Ep'] + df['Nf'])
        df['Rank'] = rankdata(df['Jaccard'], method='min')  # rank data based on coefficient
        df = df.sort_values('Rank', ascending=False)  # sort data by rank
        top = df['Rank'].unique()[:TOP]  # find top N unique values in rank
        df_top = df.loc[df['Rank'].isin(top)]  # select top N unique values in dataframe
    
        try:
            df_2 = faults_extractor(faulty_paths[idx])
            if not df['Line'].isin(df_2['Line']).any():
                raise NotFoundInTheDataFrame()
        except ParserError as pe:
            warnings.warn(f'Fault with file {faulty_paths[idx]}')
            warnings.warn(str(pe))
            failed += 1
            count_all -= 1  # we must decrease a number of all element as this one cannot be used for evaluation
            continue
        except NotFoundInTheDataFrame:
            warnings.warn(f'Fault with file {coverage_paths[idx]}')
            warnings.warn('Line from faulty csv was not found in the whole dataframe')
            failed += 1
            count_all -= 1  # we must decrease a number of all element as this one cannot be used for evaluation
            continue

        if np.sum(df_top['Line'].isin(df_2['Line'])) == df_2.shape[0]: #if we found all faults
            count += 1

    acc = count/count_all
    print(f'Accuracy to find faulty line within top {TOP} = {acc} with {failed} out of {data_size} failed files')
