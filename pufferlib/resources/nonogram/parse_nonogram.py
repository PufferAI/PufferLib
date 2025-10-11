import pandas as pd
import numpy as np
import json

EMPTY = 0

def shift_left(arr):
    result = np.full_like(arr, EMPTY)
    for i, row in enumerate(arr):
        valid = row[row != EMPTY]
        result[i, :len(valid)] = valid
    return result

def parse_nonogram(data_path):
    df = pd.read_csv(data_path)
    puzzles = []
    N = 8

    for index, row in df.iterrows():
        sizeCol = row['sizeCol']
        sizeRow = row['sizeRow']

        if max(sizeCol, sizeRow) > N:
            continue

        colClues = np.array(row['colClues'].replace('\xa0', str(EMPTY)).split(','), dtype=int)
        rowClues = np.array(row['rowClues'].replace('\xa0', str(EMPTY)).split(','), dtype=int)
        colClues = colClues.reshape((sizeCol, -1))
        rowClues = rowClues.reshape((sizeRow, -1))

        colClues = shift_left(colClues)
        colCluesPadded = np.full((N, N//2), EMPTY)
        colCluesPadded[:sizeCol, :colClues.shape[1]] = colClues

        rowClues = shift_left(rowClues)
        rowCluesPadded = np.full((N, N//2), EMPTY)
        rowCluesPadded[:sizeRow, :rowClues.shape[1]] = rowClues

        puzzle = {
            'sizeCol': int(sizeCol),
            'sizeRow': int(sizeRow),
            'title': row['title'],
            'number': int(row['number']),
            'solution': row['solution'],
            'difficulty': row['difficulty'],
            'colClues': colCluesPadded.tolist(),
            'rowClues': rowCluesPadded.tolist(),
        }

        puzzles.append(puzzle)

    return puzzles

if __name__ == '__main__':
    puzzles = parse_nonogram(data_path='/Users/eporat/Projects/PufferLib/pufferlib/resources/nonogram/nonogram.csv')
    with open('data.json', 'w') as f:
        print(len(puzzles))
        json.dump(puzzles, f)