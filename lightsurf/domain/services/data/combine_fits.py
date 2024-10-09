import logging
from pathlib import Path

import click
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from lightsurf.constants import WAVELENGTH_AIR

MODULE_PATH = Path(__file__).parents[4]


def extract_flux(star_file_path: str) -> list:
    star_spec = fits.open(star_file_path)
    flux = star_spec[1].data
    return flux


def combine_fluxes(star_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    FLUX = []
    LOADED_STARS = []
    FAILED_STARS = []
    for star_file_path in tqdm(star_paths):
        try:
            FLUX.append(extract_flux(star_file_path))
            LOADED_STARS.append(star_file_path.stem)
        except FileNotFoundError:
            logging.warning(f"File not found: {star_file_path}")
            FAILED_STARS.append(star_file_path)
            continue
        except TypeError as e:
            logging.warning(f"{e}: {star_file_path}")
            FAILED_STARS.append(star_file_path)
            continue
    star_ids = [star_path for star_path in LOADED_STARS]
    wavelength_air = [f"{wave:.2f}" for wave in WAVELENGTH_AIR]
    FLUX = pd.DataFrame(FLUX, columns=wavelength_air, index=star_ids)
    FAILED_STARS = pd.DataFrame(FAILED_STARS, columns=["FILE"])
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
def main(star_path: str | Path, output_path: str | Path) -> None:
    print(f"🔍 Finding fit files at {star_path}")
    star_paths = list(Path(star_path).glob("*.fits"))
    print(f"👉 {len(star_paths)} files found")
    print("🔀 Combining fluxes")
    FLUX, FAILED_STARS = combine_fluxes(star_paths=star_paths)
    FLUX.index.name = "FILE"
    print(f"📦 Saving combined fluxes at {output_path}")
    FLUX.to_csv(f"{output_path}/flux.csv")
    print(f"📦 Saving failed stars at {output_path}")
    FAILED_STARS.to_csv(f"{output_path}/failed_stars.csv", index=False)
    print("✅ Done")


if __name__ == "__main__":
    main()
