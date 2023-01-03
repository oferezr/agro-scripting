from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PLAIT_PATH = 'data/plait.csv'
INPUT_PATH = 'data/input.csv'
OUTPUT_PATH = 'output/'
DATA_COL_START = 3  # col to start in input
REMOVEABLE_CHARS = ['%']  # chars to remove in plait mapping
LUM_START = 101  # row that the lum section start
OD_START = 48  # row that the od section start
SAMPLES = 49  # amount of samples in the input at each section
GRAPH_NAME = {'od': 'OD', 'lum': 'LUM', 'lum_div_od': 'LUM/OD'}  # title of graph for each graph
GRAPH_YLABEL = {'od': 'OD', 'lum': 'LUM (RLU)', 'lum_div_od': 'LUM/OD'}  # ylBEL OF each graph
HOURS = [i for i in range(SAMPLES)]


def create_cell_to_idx():
    maper = dict()
    col = 0
    for r in range(ord('A'), ord('H') + 1):
        for c in range(1, 13):
            maper[chr(r) + str(c)] = col
            col += 1
    return maper


def parse_cell(cell: str):
    res = cell
    for c in REMOVEABLE_CHARS:
        res = res.replace(c, '')
    res = res.strip(' ').replace(' ', '_')
    return res


def generate_map(path: str) -> Dict[str, List[int]]:
    col_maper = create_cell_to_idx()
    mid_map = dict()
    df = pd.read_csv(path).values
    for i in range(8):
        for j in range(1, 13):
            cell = df[i][0] + str(j)
            mid_map[cell] = parse_cell(df[i][j])
    result = defaultdict(list)
    for cell, type in mid_map.items():
        result[type].append(col_maper[cell])
    for key in result.keys():
        result[key].sort()
    return dict(result)


def get_lb_avg(array: np.array, maper: Dict[str, List[int]]):
    lb_mtx = array[:, maper['LB']]
    avg = np.mean(lb_mtx)
    averages = []
    for col in maper['LB']:
        m = np.abs(np.mean(array[:, col]) - avg)
        averages.append(m)
    max_avg = max(averages) + 1
    cols = []
    for i in range(6):
        idx = np.argmin(averages)
        cols.append(maper['LB'][idx])
        averages[idx] = max_avg
    updated_lb_mtx = array[:, cols]
    return np.mean(updated_lb_mtx, axis=1)


def get_averages(array: np.array, maper: Dict[str, List[int]]) -> Dict[str, List[float]]:
    averages = defaultdict(list)
    for i in range(array.shape[0]):
        for type in maper.keys():
            if type == 'LB':
                continue
            ar = array[i, maper[type]]
            temp = np.mean(ar)
            averages[type].append(temp)
    return dict(averages)


def get_table_from_dict(avreges: Dict[str, List[float]]) -> pd.DataFrame:
    res = []
    for key, value in avreges.items():
        temp = [key] + value
        res.append(temp)
    return pd.DataFrame(np.array(res).T)


def norm_averages(avreges: Dict[str, List[float]], lb_avg: np.array) -> Dict[str, List[float]]:
    result = dict()
    for key, value in avreges.items():
        result[key] = (np.array(value) - lb_avg).tolist()
    return result


def make_graphs(avreges: Dict[str, List[float]], output_path: str, name: str):
    x = np.array(HOURS) / 2
    x = x.tolist()
    for key, value in avreges.items():
        plt.plot(x, value, label=key)
        plt.ylabel(GRAPH_YLABEL[name])
        plt.xlabel('time(h)')
    plt.title(GRAPH_NAME[name])
    plt.legend(title='Treatment')
    plt.savefig(output_path)
    plt.close()


def pipline(maper: Dict[str, List[int]], input: np.array, output_path: str, name: str):
    averages = get_averages(input, maper)
    avg_table = get_table_from_dict(averages)
    lb_avg = get_lb_avg(input, maper)
    lb_table = pd.DataFrame(np.array(['LB_AVG'] + lb_avg.tolist()).T)
    norm_avgs = norm_averages(averages, lb_avg)
    norm_table = get_table_from_dict(norm_avgs)
    s_table = pd.concat([avg_table,lb_table,norm_table],axis=1)
    s_table.to_csv(output_path+name+'.csv')
    make_graphs(norm_avgs, output_path + name + '.png',name)


def run_all(map_path: str, input_path: str, output_path: str):
    maper = generate_map(map_path)
    df = pd.read_csv(input_path).values
    od_mtx = df[OD_START:OD_START + SAMPLES, DATA_COL_START:].astype(float)
    lum_mtx = df[LUM_START:LUM_START + SAMPLES, DATA_COL_START:].astype(float)
    div_mtx = lum_mtx / od_mtx
    pipline(maper, od_mtx, output_path, 'od')
    pipline(maper, lum_mtx, output_path, 'lum')
    pipline(maper, div_mtx, output_path, 'lum_div_od')


if __name__ == '__main__':
    run_all(PLAIT_PATH, INPUT_PATH, OUTPUT_PATH)
