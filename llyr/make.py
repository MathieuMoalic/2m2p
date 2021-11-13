from typing import Optional, Dict
import os
import re
import glob
import shutil
import time

import numpy as np
import imageio

# from ._utils import get_config
from .ovf import get_ovf_parms


class Make:
    def __init__(self, llyr):
        self.llyr = llyr
        self.skip_ovf: bool
        self.ts: int
        self.force: bool
        self.out_path: str
        self.mx3_path: str
        self.logs_path: str
        self.dset_prefixes: dict

    def make(
        self,
        load_path: Optional[str] = None,
        tmax: Optional[int] = None,
        force: bool = False,
        delete_out: bool = False,
        delete_mx3: bool = False,
        skip_ovf: bool = False,
    ):
        if load_path is None:
            load_path = self.llyr.path.replace(".h5", ".out")
        self.skip_ovf = skip_ovf
        self.ts = 0
        self.force = force
        self.llyr.h5.create_h5(force)
        self.llyr.h5.add_attr("version", "0.1.5")
        self.add_paths(load_path)
        self.add_times()
        self.add_step_size()
        self.add_mx3()
        self.add_snapshots()
        self.add_logs()
        self.add_table()
        if not skip_ovf:
            self.add_dset_prefixes()
            for prefix, name in self.dset_prefixes.items():
                self.make_dset(prefix, name=name, tmax=tmax)
        if delete_out:
            shutil.rmtree(self.out_path)
        if self.mx3_path != "" and delete_mx3:
            os.remove(self.mx3_path)

    def add_paths(self, load_path: str):
        """Cleans the input string and return the path for .out folder and .mx3 file"""
        load_path = load_path.replace(".mx3", "").replace(".out", "").replace(".h5", "")

        if os.path.exists(f"{load_path}.out"):
            self.out_path = f"{load_path}.out"
        else:
            raise NameError(f"{load_path}.out not found")

        if os.path.exists(f"{load_path}.mx3"):
            self.mx3_path = f"{load_path}.mx3"
        else:
            self.mx3_path = ""

        if os.path.exists(f"{self.out_path}/slurm.logs"):
            self.logs_path = f"{self.out_path}/slurm.logs"
        elif os.path.exists(f"{self.out_path}/log.txt"):
            self.logs_path = f"{self.out_path}/log.txt"
        else:
            self.logs_path = ""

    def add_times(self):
        start_file_path = f"{self.out_path}/gui"
        stop_file_path = f"{self.out_path}/log.txt"
        if os.path.exists(start_file_path):
            self.llyr.h5.add_attr(
                "start_time",
                time.asctime(time.localtime(os.stat(start_file_path).st_mtime)),
            )
        if os.path.exists(stop_file_path):
            self.llyr.h5.add_attr(
                "stop_time",
                time.asctime(time.localtime(os.stat(stop_file_path).st_mtime)),
            )

    def add_step_size(self):
        # load one file to initialize the h5 dataset with the correct shape
        ovf_path = glob.glob(f"{self.out_path}/*.ovf")[0]
        ovf_parms = get_ovf_parms(ovf_path)
        for key in ["dx", "dy", "dz"]:
            if key not in self.llyr.attrs:
                self.llyr.h5.add_attr(key, ovf_parms[key])

    def add_mx3(self):
        """Adds the mx3 file to the f.attrs"""
        if self.mx3_path != "":
            with open(self.mx3_path, "r") as mx3:
                self.llyr.h5.add_attr("mx3", mx3.read())
        else:
            print("mx3 file not found")

    def add_snapshots(self):
        snapshots_paths = glob.glob(f"{self.out_path}/*.png") + glob.glob(
            f"{self.out_path}/*.jpg"
        )
        for p in snapshots_paths:
            snapshot = imageio.imread(p)
            name = p.split("/")[-1].replace(".png", "")
            self.llyr.h5.add_dset(snapshot, f"snapshots/{name}", force=self.force)

    def add_logs(self):
        """Adds the logs file to the f.attrs"""
        if os.path.isfile(self.logs_path):
            with open(self.logs_path, "r") as logs:
                self.llyr.h5.add_attr("logs", logs.read())
        else:
            print("logs not found")

    def add_table(self, dset_name: str = "table"):
        """Adds a the mumax table.txt file as a dataset"""
        table_path = f"{self.out_path}/table.txt"
        if os.path.isfile(table_path):
            with open(table_path, "r") as table:
                header = table.readline()
                data = np.loadtxt(table).T
            # Add dt
            if self.skip_ovf:
                times = data[0]
                dt = (times[-1] - times[0]) / (len(times) - 1)
                self.llyr.h5.add_attr("dt", dt)
            # add table data
            clean_header = [
                i.split(" (")[0].replace("# ", "") for i in header.split("\t")
            ]
            for i, h in enumerate(clean_header):
                self.llyr.h5.add_dset(data[i], f"{dset_name}/{h}", force=self.force)

    def add_dset_prefixes(self):
        """From the .out folder, get the list of prefixes, each will correspond to a different dataset"""
        paths = glob.glob(f"{self.out_path}/*.ovf")
        prefixes = list(
            {re.sub(r"_?[\d.]*.ovf", "", path.split("/")[-1]) for path in paths}
        )
        names = {}
        # prefix_to_name = get_config()
        prefix_to_name: Dict[str, str] = dict()
        for prefix in prefixes:
            names[prefix] = prefix
            for pattern, name in prefix_to_name.items():
                if re.findall(pattern, prefix.lower()):
                    names[prefix] = name
                    break
        self.dset_prefixes = names

    def make_dset(
        self,
        prefix: str,
        name: str,
        tmax: Optional[int] = None,
    ):
        """Creates a dataset from an input .out folder path and a prefix (i.e. "m00")"""
        ovf_paths = sorted(glob.glob(f"{self.out_path}/{prefix}*.ovf"))[:tmax]
        # this is to calculate dt
        if self.ts < len(ovf_paths):
            self.ts = len(ovf_paths)

        ovf_parms = get_ovf_parms(ovf_paths[-1])
        if "dt" not in self.llyr.dsets and len(ovf_paths) > 2:
            t0 = get_ovf_parms(ovf_paths[0])["t"]
            tn = ovf_parms["t"]
            dt = (tn - t0) / len(ovf_paths)
            self.llyr.h5.add_attr("dt", dt)
        dset_shape = (len(ovf_paths),) + ovf_parms["shape"]
        self.llyr.h5.load_dset(name, dset_shape, ovf_paths)
