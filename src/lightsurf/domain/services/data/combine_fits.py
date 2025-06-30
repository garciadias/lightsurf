import gc
import logging
from pathlib import Path

import click
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from lightsurf.constants import APOGEE_PARAMETERS, APOGEE_WAVELENGTH_AIR
from lightsurf.domain.services.data.download_spectra import load_star_list

MODULE_PATH = Path(__file__).parents[5]


def extract_flux(star_file_path: str, data_position: int = 1) -> list:
    hdul = fits.open(star_file_path)
    image_data = hdul[data_position].data.copy()
    hdul.close()
    gc.collect()

    return image_data


def extract_abundances(star_list_fits_file_path: str | Path) -> pd.DataFrame:
    dr = load_star_list(star_list_fits_file_path)
    abundances = [dr[var] for var in APOGEE_PARAMETERS]
    abundances = pd.DataFrame(
        abundances, index=APOGEE_PARAMETERS, columns=dr["APOGEE_ID"]
    ).T
    abundances.index.name = "FILE"
    abundances.reset_index(inplace=True)
    abundances.drop_duplicates(subset="FILE", inplace=True, keep="first")
    abundances["FILE"] = "aspcapStar-dr17-" + abundances["FILE"]
    abundances.set_index("FILE", inplace=True)
    return abundances


def combine_fluxes(
    star_paths: list[Path], data_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_position = 1 if data_type == "spectrum" else 3
    FLUX = []
    LOADED_STARS = []
    FAILED_STARS = []
    for star_file_path in tqdm(star_paths, desc="🔀 Combining fluxes"):
        try:
            FLUX.append(extract_flux(star_file_path, data_position))
            LOADED_STARS.append(star_file_path.stem)
        except FileNotFoundError:
            logging.warning(f"File not found: {star_file_path}")
            FAILED_STARS.append(star_file_path)
            continue
        except TypeError as e:
            logging.warning(f"{e}: {star_file_path}")
            FAILED_STARS.append(star_file_path)
            continue
        except IndexError as e:
            logging.warning(f"{e}: {star_file_path}")
            FAILED_STARS.append(star_file_path)
            continue
    star_ids = [star_path for star_path in LOADED_STARS]
    wavelength_air = [f"{wave:.2f}" for wave in APOGEE_WAVELENGTH_AIR]
    print(f"👉 {len(FLUX)} stars loaded")
    print("🧮 Converting fluxes to DataFrame")
    FLUX = pd.DataFrame(FLUX, columns=wavelength_air, index=star_ids)
    FAILED_STARS = pd.DataFrame(FAILED_STARS, columns=["FILE"])
    print(f"📊 {FLUX.shape} fluxes combined")
    return FLUX, FAILED_STARS


@click.command()
@click.option(
    "--star-path",
    default=f"{MODULE_PATH}/data/raw_data/apstar/",
    help="Path to star files",
)
@click.option(
    "--output-path",
    default=f"{MODULE_PATH}/data/raw_data/",
    help="Path to save the combined fluxes",
)
@click.option(
    "--data_type",
    default="spectrum",
    help="Type of spectral data to extract. If 'spectrum' get the first \
         spectrum availiable, if 'model' gets the best fit model.",
)
def main(star_path: str | Path, output_path: str | Path, data_type: str) -> None:
    print(f"🔍 Finding fit files at {star_path}")
    star_paths = list(Path(star_path).glob("*.fits"))
    print(f"👉 {len(star_paths)} files found")
    FLUX, FAILED_STARS = combine_fluxes(star_paths=star_paths, data_type=data_type)
    FLUX.index.name = "FILE"
    print("📤 Extracting abundances")
    ABUNDANCES = extract_abundances(
        star_list_fits_file_path=f"{MODULE_PATH}/data/mdwarfs_DR17.fits"
    )
    print(f"📦 Saving failed stars at {output_path}")
    FAILED_STARS.to_csv(f"{output_path}/failed_stars.csv", index=False)
    # Combine fluxes and abundances
    FLUX = pd.concat([FLUX, ABUNDANCES], axis=1)
    print(FLUX.index[:5], ABUNDANCES.index[:5])
    FLUX.dropna(subset=ABUNDANCES.columns, inplace=True, how="all")
    print(f"📊 {FLUX.shape} fluxes and abundances combined")
    print(f"📦 Saving combined fluxes and abundances at {output_path}")
    if data_type == "spectrum":
        FLUX.round(4).to_csv(f"{output_path}/flux_abundances.csv", float_format="%.4f")
    else:
        FLUX.round(4).to_csv(
            f"{output_path}/best_fit_flux_abundances.csv", float_format="%.4f"
        )
    print("✅ Done")


if __name__ == "__main__":
    main()
