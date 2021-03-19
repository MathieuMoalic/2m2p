import numpy as np
from glob import glob
import os
import re
import h5py  # type: ignore
import multiprocessing as mp
from typing import *
from typing import Match


class Make:
    path: str
    name: str

    def _load_ovf(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            for _ in range(28):
                next(f)
            try:
                arr: np.ndarray = np.fromfile(f, "<f4", count=self._count)[1:].reshape(self._ovf_shape)
            except ValueError as e:
                print("ovf files are corrupted")
                return e
        return arr

    def _get_ovf_shape(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            while True:
                line = f.readline().strip().decode("ASCII")
                if "valuedim" in line:
                    c = int(line.split(" ")[-1])
                if "xnodes" in line:
                    x = int(line.split(" ")[-1])
                if "ynodes" in line:
                    y = int(line.split(" ")[-1])
                if "znodes" in line:
                    z = int(line.split(" ")[-1])
                    break
        return (z,y,x,c)

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
                # print(line,key_match,value_match,key_re,value_re)
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

            lines = f.attrs["mx3"]
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

    def _get_paths(self,load_path):
        if load_path[-3:] in ['mx3','out']:
            load_path = load_path[:-4]
        out_path = f"{load_path}.out"
        mx3_path = f"{load_path}.mx3"
        return out_path, mx3_path

    def add_mx3(self,mx3_path):
        print(mx3_path)
        if os.path.isfile(mx3_path):
            with open(mx3_path, "r") as mx3:
                with h5py.File(self.h5_path, "a") as f:
                    f.attrs["mx3"] = mx3.read()
        else:
            print(f"{mx3_path} not found")

    def _create_h5(self):
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

    def _get_dset_prefixes(self,out_path):
        prefixes = [i.split("/")[-1].replace("_000000.ovf","") for i in glob(f"{out_path}/*00000.ovf")]
        if os.path.isfile(f"{out_path}/stable.ovf"):
            prefixes.append("stable")
        return prefixes

    def _get_dset_name(self,prefix):
        # this func replaces common prefixes with more readable dset names
        common_prefix_to_name = (("m_zrange4","ND"),("m_zrange2","WG"),("stable","stable"))
        for i in common_prefix_to_name:
            if i[0] in prefix:
                return i[1]
        return prefix

    def add_dset(self, out_path, prefix, name=None, tmax=None, force=False):
        ovf_paths = sorted(glob(f"{out_path}/{prefix}*.ovf"))[:tmax]
        # load one file to initialize the h5 dataset with the correct shape
        self._ovf_shape = self._get_ovf_shape(ovf_paths[0])
        dset_shape = ((len(ovf_paths),)+self._ovf_shape)
        # number of bytes in the data used in self._load_ovf, (+1 is for the security number of ovf) 
        self._count = self._ovf_shape[0] * self._ovf_shape[1] * self._ovf_shape[2] * self._ovf_shape[3] + 1
        if name is None:
            name = self._get_dset_name(prefix)
        
        with h5py.File(self.h5_path, "a") as f:
            if force and name in list(f.keys()):
                del f[name]
            dset = f.create_dataset(name, dset_shape, np.float32)
            pool = mp.Pool(processes=int(mp.cpu_count() - 1))
            for i, data in enumerate(pool.imap(self._load_ovf, ovf_paths)):
                dset[i] = data
            pool.close()
            pool.join()

    def add_np_dset(self,arr,name,force=False):
        if force and name in list(f.keys()):
            del f[name]
        with h5py.File(self.h5_path, "a") as f:
            f.create_dataset(name,data=arr)

    def make(self, load_path: str, tmax=None) -> None:
        # automatically parse the load_path and will create datasets etc ..
        self._create_h5()
        out_path, mx3_path = self._get_paths(load_path)
        self.add_table(f"{out_path}/table.txt")
        self.add_mx3(mx3_path)
        self.parse_script()
        dset_prefixes = self._get_dset_prefixes(out_path)
        for dset_prefix in dset_prefixes:
            self.add_dset(out_path,dset_prefix,tmax=tmax)
