import asyncio
from pathlib import Path

import click
import tqdm
from astropy.io import fits

MODULE_PATH = Path(__file__).parents[5]


async def download(link: str, out: str | Path) -> Path:
    output_file_path = Path(f"{out}/{link.split('/')[-1]}")
    if not output_file_path.exists():
        process = await asyncio.create_subprocess_shell(
            f"wget {link} --http-user=sdss --http-passwd=2.5-meters -P {out}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Download failed: {stderr.decode()}")
    return output_file_path


def load_star_list(file_path: str | Path) -> list:
    dr = fits.open(file_path)
    dr = dr[1].data
    return dr


async def download_star(star_data: dict, output_path: str | Path) -> None:
    Path(output_path).mkdir(parents=True, exist_ok=True)
    telescope = str(star_data["telescope"])
    # SDSS-V groups apStar files by coarse field = healpix // 256 (Nside=16).
    # The allStar-1.3 'file' column holds the exact filename (incl. MJD).
    field = str(int(star_data["healpix"]) // 256)
    filename = str(star_data["file"])
    base_link = "https://dr19.sdss.org/sas/dr19/spectro/apogee/redux/1.3/stars/"
    link = f"{base_link}{telescope}/{field}/{filename}"
    await download(link, output_path)


async def download_in_batches(
    star_data_list: list, batch_size: int, output_path: str | Path
) -> None:
    n_stars = len(star_data_list)
    for i in tqdm.tqdm(range(0, n_stars, batch_size), desc="🌐 Downloading spectra"):
        batch = star_data_list[i : i + batch_size]
        tasks = [download_star(star_data, output_path) for star_data in batch]
        await asyncio.gather(*tasks)


@click.command()
@click.option(
    "--star_list_fits_file_path",
    help="Path to the FITS file containing the list of stars",
    default=f"{MODULE_PATH}/data/allStar-1.3-apo25m.fits",
)
@click.option(
    "--output_path",
    help="Path to the folder where the spectra will be saved",
    default=f"{MODULE_PATH}/data/raw_data/apstar",
)
@click.option("--n_workers", help="Number of workers", default=50)
def download_spectra(
    star_list_fits_file_path: str | Path, output_path: str | Path, n_workers: int = 50
) -> None:
    print("📦 Loading list of stars")
    star_list_fits_file_path = Path(star_list_fits_file_path)
    dr = load_star_list(star_list_fits_file_path)
    print(f"👉 {len(dr)} stars will be downloaded and saved at {output_path}")
    asyncio.run(download_in_batches(dr, n_workers, output_path))
    print("✅ Done")


if __name__ == "__main__":
    download_spectra()
