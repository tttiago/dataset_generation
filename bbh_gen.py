import argparse
import warnings
from typing import Tuple

import h5py
import numpy as np
from pycbc.waveform import get_td_waveform
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Generator:
    def __init__(
        self,
        dataset_name: str = "BBH_dataset",
        num_waveforms: int = 1e4,
        approximant: str = "IMRPhenomPv2_NRTidalv2",
        mass_min: float = 1.0,
        mass_max: float = 2.0,
        distance: float = 100.0,
        inclination: float = 0.0,
        sample_rate: int = 8192,
        freq_lower: int = 100,
    ):
        self.dataset_name = dataset_name
        self.num_waveforms = num_waveforms
        self.approximant = approximant
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.distance = distance
        self.inclination = inclination
        self.sample_rate = sample_rate
        self.freq_lower = freq_lower

        self.hps, self.hcs = [], []
        self.configs = {
            k: []
            for k in (
                "mass1",
                "mass2",
                "distance",
                "inclination",
                "freq_lower",
                "sample_rate",
                "approximant",
            )
            # for k in ()
        }

    @property
    def configuration(self) -> dict:
        rng = np.random.default_rng()
        mass1 = np.round(rng.uniform(self.mass_min, self.mass_max), 3)
        mass2 = np.round(rng.uniform(self.mass_min, mass1), 3)

        config = {
            "mass1": mass1,
            "mass2": mass2,
            "distance": self.distance,
            "inclination": self.inclination,
            "approximant": self.approximant,
            "freq_lower": self.freq_lower,
            "sample_rate": self.sample_rate,
        }
        return config

    def _save_sample(self, config, hp, hc):
        hp = np.array(hp)[-1024:]
        hc = np.array(hc)[-1024:]
        self.hps.append(hp)
        self.hcs.append(hc)
        for k in self.configs:
            self.configs[k].append(config[k])

    @staticmethod
    def gen_data_strain(config: dict) -> Tuple[list]:
        hp, hc = get_td_waveform(
            approximant=config["approximant"],
            mass1=config["mass1"],
            mass2=config["mass2"],
            distance=config["distance"],
            inclination=config["inclination"],
            f_lower=config["freq_lower"],
            delta_t=1.0 / config["sample_rate"],
        )
        return hp, hc

    def _create_samples(self, idx):
        success = False
        # Generate a waveform, changing the configuration if an error occurs.
        while not success:
            config = self.configuration
            try:
                hp, hc = self.gen_data_strain(config)
                success = True
            except RuntimeError:
                continue
            except Exception as exception:
                print(f"\n\nException: {type(exception).__name__}\n\n")

        # Save data.
        self._save_sample(config, hp, hc)

    def run_gen(self) -> None:
        print("Generating dataset...")
        for i in tqdm(range(1, self.num_waveforms + 1)):
            self._create_samples(idx=i)

    def save_hdf5(self) -> None:
        with h5py.File(f"{self.dataset_name}.hdf5", "w") as f:
            f.create_dataset("h_plus", data=self.hps, compression="gzip", chunks=True)
            f.create_dataset("h_cross", data=self.hcs, compression="gzip", chunks=True)
            for config_name, config_list in self.configs.items():
                f.attrs.create(config_name, config_list)


def create_parser():
    parser = argparse.ArgumentParser(description="Generate GW data.")
    parser.add_argument("-n", "--num_waveforms", type=int, default=int(1e4))
    parser.add_argument("--dataset_name", type=str, default="BBH_dataset")
    parser.add_argument("--approximant", type=str, default="IMRPhenomPv2")
    parser.add_argument("--mass_min", type=float, default=1.0)
    parser.add_argument("--mass_max", type=float, default=2.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--inclination", type=float, default=0.0)
    parser.add_argument("--sample_rate", type=int, default=1024)
    parser.add_argument("--freq_lower", type=float, default=20.0)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    gen = Generator(**args_dict)

    gen.run_gen()
    gen.save_hdf5()
