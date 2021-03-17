import numpy as np
from glob import glob
import os
import re
import h5py  # type: ignore
import multiprocessing as mp
from typing import *


class Make:
    path: str
    name: str

    def load_ovf(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            dims: np.ndarray = np.array([0, 0, 0, 0])
            while True:
                line = f.readline().strip().decode("ASCII")
                if "valuedim" in line:
                    dims[3] = int(line.split(" ")[-1])
                if "xnodes" in line:
                    dims[2] = int(line.split(" ")[-1])
                if "ynodes" in line:
                    dims[1] = int(line.split(" ")[-1])
                if "znodes" in line:
                    dims[0] = int(line.split(" ")[-1])
                if "Begin: Data" in line:
                    break
            count: int = int(dims[0] * dims[1] * dims[2] * dims[3] + 1)
            arr: np.ndarray = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
        return arr

    def parse_script(self) -> None:
        with h5py.File(self.h5_path, "a") as f:
            line: str
            key: str
            value: float
            key_match: Optional[Match[str]]
            value_match: Optional[Match[str]]
            region_match: Optional[Match[str]]

            def get_attribute(key_re: str, value_re: str, region_re: str=None) -> None:
                key_match = re.search(key_re, line)
                value_match = re.search(value_re, line)
                print(line,key_match,value_match,key_re,value_re)
                if region_re is not None:
                    region_match = re.search(region_re, line)

                    if isinstance(key_match, Match) and isinstance(value_match, Match) and isinstance(region_match, Match):
                        key = region_match.group().split("_")[0] + "_" + key_match.group().replace(":=","").replace(" ","")
                        value = float(value_match.group())
                        f.attrs[key] = value
                        return True

                elif isinstance(key_match, Match) and isinstance(value_match, Match):
                    key = key_match.group().replace(":=","").replace(" ","")
                    value = float(value_match.group())
                    f.attrs[key] = value
                    return True
                return False

        try:
            lines = f.attrs["script"]
            lines = lines.split("\n")
            for line in lines:
                line = line.lower()
                # matches magnetic parameters
                if "setregion" in line and "anisu" not in line:
                    if get_attribute(r"\w+", r"\d+\.?\d*e?-?\d*"):
                        continue
                # matches cell dimensions
                if get_attribute(r"[d][xyz]", r"\d+.?\d*e-\d+"):
                    continue
                # matches amplitude
                if get_attribute(r"amps *:=", r"\d+.?\d*e?-?\d*"):
                    continue
                # matches frequency
                if get_attribute(r"^[[f]|[f_cut]] *:=", r"\d+.?\d*e?-?\d*"):
                    continue
        except:
            print("Couldn't parse script")

    def add_table(self, table_path, name="table"):
        if os.path.isfile(table_path):
            with open(table_path, "r") as table:
                header: str = table.readline()
                data: np.ndarray = np.loadtxt(table).T
                dt: float = (data[0, -1] - data[0, 0]) / (data.shape[1] - 1)
            with h5py.File(self.h5_path, "a") as f:
                tableds = f.create_dataset(name, data=data)
                tableds.attrs["header"] = header
                f.attrs["dt"] = dt
        else:
            print("table.txt not found")

    def get_paths(self,loadpath):
        if loadpath[-3:] in ['mx3','out']:
            loadpath = loadpath[:-4]
            out_path = f"{loadpath}.out"
            mx3_path = f"{loadpath}.mx3"
        return out_path, mx3_path

    def add_mx3(self,mx3_path):
        if os.path.isfile(mx3_path):
            with open(mx3_path, "r") as script:
                with h5py.File(self.h5_path, "a") as f:
                    f.attrs["script"] = script.read()
        else:
            print(f"{mx3_path} not found")

    def create_h5(self):
        if self.force:
            with h5py.File(self.h5_path, "w") as f:
                pass
        else:
            if os.path.isfile(self.h5_path):
                input_string: str = input(
                    f"{self.h5_path} already exists, overwrite it [y/n]?"
                )
                if input_string.lower() in ["y", "yes"]:
                    with h5py.File(self.h5_path, "w") as f:
                        pass
                else:
                    return

    def get_dset_prefixes(self,out_path):
        prefixes = [i.split("/")[-1].replace("_000000.ovf","") for i in glob(f"{out_path}/*00000.ovf")]
        if os.path.isfile(f"{out_path}/stable.ovf"):
            prefixes.append("stable")
        return prefixes

    def add_dset(self, out_path, prefix, name=None, tmax=None force=False):
        common_prefix_to_name_dict = {"zrange4":"ND","zrange2":"WG","stable":"stable"}
        ovf_paths = sorted(glob(f"{out_path}/{prefix}*.ovf"))[:tmax]
        # load one file to initialize the h5 dataset with the correct shape
        dset_shape = (len(ovf_paths),) + self.load_ovf(ovf_paths[0]).shape
        name = common_prefix_to_name_dict.get(prefix[:9],prefix)
        with h5py.File(self.h5_path, "a") as f:
            if force and name in list(f.keys):
                del f[name]
            dset = f.create_dataset(name, dset_shape, np.float32)
            pool = mp.Pool(processes=int(mp.cpu_count() - 1))
            for i, data in enumerate(pool.imap(self.load_ovf, ovf_paths)):
                dset[i] = data
            pool.close()
            pool.join()
    
    def add_np_dset(self,arr,name,force=False):
        if force and name in list(f.keys):
            del f[name]
        with h5py.File(self.h5_path, "a") as f:
            f.create_dataset(name,data=arr)

    def make(self, loadpath: str) -> None:
        # automatically parse the loadpath and will create datasets etc ..
        
        self.create_h5()
        out_path, mx3_path = self.get_paths(loadpath)
        self.add_table(f"{out_path}/table.txt")
        self.add_mx3(mx3_path)
        self.parse_script()
        dset_prefixes = self.get_dset_prefixes(out_path)
        for dset_prefix in dset_prefixes:
            self.add_dset(out_path,dset_prefix)
