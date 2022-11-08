import pandas as pd
import re
import numpy as np


def faults_extractor(filename: str) -> pd.DataFrame:

    with open(filename, 'r') as f:
        header = f.readline()
        header = list(filter(None, re.split(',|\n', header)))
        data = f.read()
        data = list(filter(lambda x: (x not in ['', ',"']), re.split('(,"|,F|,T)', data)))
        data_len = np.ceil(len(data)/3).astype(int)
        data = np.array(data)
        data = np.reshape(data, (data_len, 3))

        for idx in range(data_len):
            data[idx, 0] = data[idx, 0].replace("\n", "")
            data[idx, 1] = data[idx, 1].replace("\"", "")
            data[idx, 2] = data[idx, 2].replace(",", "")

        df = pd.DataFrame(data, columns=header)
        return df


if __name__ == '__main__':
    path = './eval-data/eval/chart14.faulty-lines.csv'
    print(faults_extractor(path))