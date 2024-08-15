from typing import Dict, List
import pandas as pd


class IORecorder:
    def __init__(self):
        self.dataset = []

    def append_element(self,
                       gl_snapshot,
                       dx_list: List,
                       t: float):
        x = gl_snapshot["x"]
        a = gl_snapshot["a"]
        d = gl_snapshot["d"]
        p = gl_snapshot["p"]
        u = gl_snapshot["u"]

        x = {f"x_{k}": v for k, v in x.items()}
        a = {f"a_{k}": v for k, v in a.items()}
        d = {f"d_{k}": v for k, v in d.items()}
        p = {f"p_{k}": v for k, v in p.items()}
        u = {f"u_{k}": v for k, v in u.items()}

        dx_dict = {f"dx_{i}": dx for i, dx in enumerate(dx_list)}

        element = {**x, **a, **d, **p, **u}
        element = {'t': t, **element, **dx_dict}

        self.dataset.append(element)

    def save_dataset(self, file_path):
        df = pd.DataFrame(self.dataset)
        df.to_csv(file_path, index=False)

    def clear_dataset(self):
        self.dataset = []
