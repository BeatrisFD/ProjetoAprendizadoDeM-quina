import pandas as pd
import os

def merge_creditcard_files():

    base_path = '../data'

    files = [
        'creditcard_part_1.csv',
        'creditcard_part_2.csv',
        'creditcard_part_3.csv',
        'creditcard_part_4.csv',
        'creditcard_part_5.csv',
        'creditcard_part_6.csv'
    ]

    dfs = []

    for file in files:

        path = os.path.join(base_path, file)

        df = pd.read_csv(path)

        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    output_path = os.path.join(base_path, 'creditcard.csv')

    merged_df.to_csv(output_path, index=False)

    print('Arquivo creditcard.csv criado com sucesso!')


if __name__ == '__main__':
    merge_creditcard_files()