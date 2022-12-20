from collections import defaultdict
from typing import Dict, List

import pandas as pd
import numpy as np

MAPPING_PATH = 'data/plait1.csv'
INPUT_PATH = 'data/input.csv'
OUTPUT_PATH = 'data/output.csv'
DATA_COL_START = 3
REMOVEABLE_CHARS = ['%', '(1)', '(2)', '(3)']
INPUT_DATA_START = 48  # 101
SAMPLES = 49


def create_cell_to_idx():
    maper = dict()
    col = DATA_COL_START
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


def generate_map(path: str):
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
    avg = np.mean(lb_mtx.astype(float))
    averages = []
    for col in maper['LB']:
        m = np.abs(np.mean(array[:, col].astype(float)) - avg)
        averages.append(m)
    max_avg = max(averages) + 1
    cols = []
    for i in range(6):
        idx = np.argmin(averages)
        cols.append(idx)
        averages[idx] = max_avg
    return np.mean(array[:, cols].astype(float))


def get_averages(array: np.array, maper: Dict[str, List[int]]) -> Dict[str, List[float]]:
    averages = defaultdict(list)
    for i in range(array.shape[0]):
        for type in maper.keys():
            if type == 'LB':
                continue
            ar = array[i, maper[type]].astype(float)
            temp = np.mean(ar)
            averages[type].append(temp)
    return dict(averages)


def pipline(map_path: str, input_path: str, output_path: str):
    maper = generate_map(map_path)
    print(maper)
    df = pd.read_csv(input_path).values[INPUT_DATA_START:INPUT_DATA_START + SAMPLES]
    averages = get_averages(df, maper)
    lb_avg = get_lb_avg(df, maper)
    x=0


if __name__ == '__main__':
    pipline(MAPPING_PATH, INPUT_PATH, OUTPUT_PATH)
