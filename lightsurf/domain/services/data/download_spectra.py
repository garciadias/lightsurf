import asyncio
from pathlib import Path

import tqdm
from astropy.io import fits

MODULE_PATH = Path(__file__).parents[4]


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
    telescope = str(star_data["TELESCOPE"])
    field = str(star_data["FIELD"])
    star = str(star_data["APOGEE_ID"])
    base_link = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/"
    link = f"{base_link}{telescope}/{field}/aspcapStar-dr17-{star}.fits"
    await download(link, output_path)


async def download_in_batches(
    star_data_list: list, batch_size: int, output_path: str | Path
) -> None:
    n_stars = len(star_data_list)
    for i in tqdm.tqdm(range(0, n_stars, batch_size), desc="🌐 Downloading spectra"):
        batch = star_data_list[i : i + batch_size]
        tasks = [download_star(star_data, output_path) for star_data in batch]
        await asyncio.gather(*tasks)


def main(
    star_list_fits_file_path: str | Path, output_path: str | Path, n_workers: int = 50
) -> None:
    print("📦 Loading list of stars")
    star_list_fits_file_path = Path(star_list_fits_file_path)
    dr = load_star_list(star_list_fits_file_path)
    print(f"👉 {len(dr)} stars will be downloaded and saved at {output_path}")
    asyncio.run(download_in_batches(dr, n_workers, output_path))
    print("✅ Done")


if __name__ == "__main__":
    main(
        star_list_fits_file_path=f"{MODULE_PATH}/data/mdwarfs_DR17",
        output_path=f"{MODULE_PATH}/data/raw_data/apstar",
    )
