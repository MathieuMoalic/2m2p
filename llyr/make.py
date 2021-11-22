import os

import numpy as np


def add_table(llyr, dset_name: str = "table"):
    """Adds a the mumax table.txt file as a dataset"""
    table_path = f"{llyr.apath}/table.txt"
    if os.path.isfile(table_path):
        with open(table_path, "r") as table:
            header = table.readline()
            data = np.loadtxt(table).T
        clean_header = [i.split(" (")[0].replace("# ", "") for i in header.split("\t")]
        grouped_headers = {i[:-1] for i in clean_header}
        groups = []
        for gh in grouped_headers:
            if (
                f"{gh}x" in clean_header
                and f"{gh}y" in clean_header
                and f"{gh}z" in clean_header
            ):
                groups.append(gh)
        groups = sorted(groups, key=len, reverse=True)
        for g in groups:
            q = []
            for c in ["x", "y", "z"]:
                i = [
                    x for x in range(len(clean_header)) if f"{g}{c}" in clean_header[x]
                ][0]
                q.append(data[i])
                clean_header = np.delete(clean_header, i)
                data = np.delete(data, i, axis=0)
            if f"{dset_name}/{g}" in llyr.z:
                del llyr.z[f"{dset_name}/{g}"]
            llyr.z.create_dataset(
                f"{dset_name}/{g}", data=np.array(q, dtype=np.float32).T
            )
        for i, h in enumerate(clean_header):
            if f"{dset_name}/{h}" in llyr.z:
                del llyr.z[f"{dset_name}/{h}"]
            llyr.z.create_dataset(f"{dset_name}/{h}", data=data[i])
