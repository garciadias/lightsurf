import os
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
from astropy.io import fits

from lightsurf.constants import WAVELENGTH_AIR

MODULE_PATH = Path(__file__).parents[4]


def download(link, out):
    print(f"Downloading {link} to {out}")
    output_file_path = Path(f"{out}/{link.split('/')[-1]}")
    if not output_file_path.exists():
        os.system(f"wget {link} --http-user=sdss --http-passwd=2.5-meters -P {out}")
    return output_file_path


def load_star_list(file_path: str):
    dr = fits.open(file_path)
    dr = dr[1].data
    return dr


def download_star(star_data, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    telescope = str(star_data["TELESCOPE"])
    field = str(star_data["FIELD"])
    star = str(star_data["APOGEE_ID"])
    base_link = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/"
    link = f"{base_link}{telescope}/{field}/aspcapStar-dr17-{star}.fits"
    downloaded = download(link, output_path)
    return downloaded


def extract_flux(star_file_path: str):
    star_spec = fits.open(star_file_path)
    flux = star_spec[1].data
    return flux


def main(star_list_fits_file_path: str, output_path: str):
    print("Loading files")
    star_list_fits_file_path = Path(star_list_fits_file_path)
    dr = load_star_list(star_list_fits_file_path)

    print(
        "This code download from SAS ASPCAP normalized spectra for\n"
        f"{len(dr)} stars on {star_list_fits_file_path.stem}"
    )

    FLUX = []
    DOWNLOADED_STARS = []
    FAIL = []
    for star_data in tqdm.tqdm(dr[:10]):
        saved_file_path = download_star(star_data, output_path)
        if saved_file_path.exists():
            DOWNLOADED_STARS.append(saved_file_path)
            FLUX.append(extract_flux(saved_file_path))
        else:
            FAIL.append(saved_file_path)
    print(f"Downloaded {len(DOWNLOADED_STARS)} stars")
    print(f"Failed to download {len(FAIL)} stars")
    FLUX = np.array(FLUX)
    print("Saving flux")
    star_ids = [star_path.stem for star_path in DOWNLOADED_STARS]
    FLUX = pd.DataFrame(FLUX, columns=WAVELENGTH_AIR, index=star_ids)
    FLUX.index.name = "APOGEE_ID"
    FLUX.to_csv(f"{output_path}/flux.csv")
    print("Flux saved")
    print("Done")


if __name__ == "__main__":
    main(
        star_list_fits_file_path=f"{MODULE_PATH}/data/mdwarfs_DR17",
        output_path=f"{MODULE_PATH}/data/raw_data/apstar",
    )
