import numpy as np
from glob import glob
import os
import re
import h5py
import multiprocessing as mp


class Make:
    def load_ovf(self, path: str):
        with open(path, "rb") as f:
            dims = np.array([0, 0, 0, 0])
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
            count = int(dims[0] * dims[1] * dims[2] * dims[3] + 1)
            arr = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
        return arr

    def parse_script(self):
        with h5py.File(self.path, "a") as f:
            for line in f.attrs["script"].split("\n"):
                line = line.lower()
                if re.search(r"[d][xyz] +:=", line) is not None:
                    key = re.search(r"[d][xyz]", line).group()
                    value = float(re.search(r"[\d.e-]+", line).group())
                    f.attrs[key] = value

                if "setregion" in line.lower() and "anisu" not in line:
                    try:
                        key = (
                            re.search(r"\w+_region", line).group().split("_")[0]
                            + "_"
                            + re.search(r"\w+", line).group()
                        )
                        value = float(re.search(r"\d+\.?\d*e?-?\d*", line[4:]).group())
                        f.attrs[key] = value
                    except:
                        pass

                if re.search(r"amps +:=", line) is not None:
                    f.attrs["amps"] = float(re.search(r"[\d.e-]+", line).group())

                if re.search(r"f +:=", line) is not None:
                    f.attrs["f"] = float(re.search(r"[\d.e-]+", line).group())

    def make(self, loadpath: str):
        if os.path.isfile(self.path):
            c = input(f"{self.path} already exists, overwrite it [y/n]?")
            if c.lower() in ["y", "yes"]:
                with h5py.File(self.path, "w") as f:
                    pass
            else:
                return

        with h5py.File(self.path, "a") as f:
            # save script
            with open(f"{loadpath}.mx3", "r") as script:
                f.attrs["script"] = script.read()

            # save table.txt
            if os.path.isfile(f"{loadpath}.out/table.txt"):
                with open(f"{loadpath}.out/table.txt", "r") as table:
                    header = table.readline()
                    tab = np.loadtxt(table).T
                    tableds = f.create_dataset("table", data=tab)
                    tableds.attrs["header"] = header
                    dt = (tab[0, -1] - tab[0, 0]) / (tab.shape[1] - 1)
                    f.attrs["dt"] = dt

            # get names of datasets
            heads = set(
                [os.path.basename(x)[:-8] for x in glob(f"{loadpath}.out/*.ovf")]
            )
            dset_names = []
            for i, head in enumerate(heads):
                if "zrange4" in head:
                    dset_names.append("ND")
                elif "zrange2" in head:
                    dset_names.append("WG")
                elif "st" in head:
                    dset_names.append("stable")
                else:
                    input_name = input(f"{head} :")
                    if input_name == "del":
                        del heads[i]
                    else:
                        dset_names.append(input_name)
            print(f"{self.name} is saving the datasets :", end="")
            for dset_name in dset_names:
                print(dset_name, end="")

            # get list of ovf files and shapes
            ovf_list = []
            ovf_shape = []
            for head in heads:
                L = glob(f"{loadpath}.out/{head}*.ovf")
                ovf_list.append(L)
                shape = self.load_ovf(L[0]).shape
                ovf_shape.append((len(L),) + shape)

            # save datasets
            for ovfs, shape, dset_name in zip(ovf_list, ovf_shape, dset_names):
                dset = f.create_dataset(dset_name, shape, np.float32)
                pool = mp.Pool(processes=int(mp.cpu_count() - 1))
                for i, d in enumerate(pool.imap(self.load_ovf, ovfs)):
                    dset[i] = d
                pool.close()
                pool.join()

            self.parse_script()
