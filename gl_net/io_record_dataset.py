from itertools import groupby
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import torch
from torch.utils.data import Dataset, DataLoader


def create_output_vector(row: Series | Dict,
                         output_columns: List[str]) -> np.ndarray:
    output_vector = np.array([row[column] for column in output_columns])
    return output_vector


def create_input_vectors(row: Series | Dict,
                         input_columns_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    required_keys = ['x', 'a', 'u', 'd', 'p']
    for key in required_keys:
        assert key in input_columns_dict, f'input_columns_dict must contain a key "{key}"'

    x = np.array([row[column] for column in input_columns_dict['x']])
    a = np.array([row[column] for column in input_columns_dict['a']])
    u = np.array([row[column] for column in input_columns_dict['u']])
    d = np.array([row[column] for column in input_columns_dict['d']])
    p = np.array([row[column] for column in input_columns_dict['p']])

    return {
        'x': x,
        'a': a,
        'u': u,
        'd': d,
        'p': p,
    }


class IODataset(Dataset):
    def __init__(self, io_record_path: str | Path):
        self.io_record_path = io_record_path
        self.io_df = self._setup_dataset(self.io_record_path)
        self.input_columns_dict, self.output_columns, self.time_column = self._get_input_output_columns(self.io_df)

    def _setup_dataset(self, io_record_path) -> DataFrame:
        io_df = pd.read_csv(io_record_path)

        nan_columns = io_df.columns[io_df.isna().any()].tolist()
        assert len(nan_columns) == 0

        # p_lambdaShScrPer is initialized with 'inf' value - so removing it...
        inf_columns = io_df.columns[np.isinf(io_df).any()].tolist()
        assert (len(inf_columns) == 1 and 'p_lambdaShScrPer' in inf_columns) or len(inf_columns) == 0
        io_df.drop('p_lambdaShScrPer', axis=1, inplace=True)

        # TODO: @gsoykan - we need to do normalization and rescaling of the inputs somehow?
        #  maybe at least for some of it - like physical limits?

        return io_df

    def _get_input_output_columns(self, io_df: DataFrame) -> Tuple[
        Dict[str, List[str]],
        List[str],
        List[str]]:
        column_names = list(io_df.columns)

        column_groups = {}

        for key, group in groupby(column_names, lambda x: x.split('_')[0]):
            column_groups[key] = list(group)

        output_columns = column_groups.pop('dx')
        time_column = column_groups.pop('t')
        input_columns_dict = column_groups

        return input_columns_dict, output_columns, time_column

    def __len__(self):
        return len(self.io_df)

    def __getitem__(self, idx):
        row = self.io_df.iloc[idx]

        # (28,)
        output_vector = create_output_vector(row, self.output_columns)
        output_vector = torch.from_numpy(output_vector).to(torch.float32)

        # {'a': (293,), 'd': (10,), 'p': (254,), 'u': (11,), 'x': (28,)}
        input_vector_dict = create_input_vectors(row, self.input_columns_dict)
        input_vector_dict = {k: torch.from_numpy(v).to(torch.float32) for k, v in input_vector_dict.items()}

        return input_vector_dict, output_vector


if __name__ == '__main__':
    io_record_csv_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLightPlus/data/io_records/20240819_115758/io_record_step_0.csv"

    io_dataset = IODataset(io_record_csv_path)

    # for i, (inputs, output) in enumerate(io_dataset):
    #     print(i, inputs, output)
    #
    #     if i == 3:
    #         break

    dataloader = DataLoader(io_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        torch.save(sample_batched, 'io_batched_instance.pt')
        print(i_batch,
              sample_batched[0]['x'].size(),
              sample_batched[1].size())

        if i_batch == 0:
            break