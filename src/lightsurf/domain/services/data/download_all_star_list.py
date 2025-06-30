import os
from pathlib import Path

import click

MODULE_PATH = Path(__file__).parents[5]


def download_file_all_star(file_url, destination):
    output_file = f"{MODULE_PATH}/{destination}"
    if not Path(output_file).exists():
        print(f"🌐 Downloading file from {file_url}")
        os.system(f"wget --no-check-certificate '{file_url}' -O {output_file}")
        print(f"🙏 Downloaded file to {destination}")
    else:
        print(f"⚠ File already exists at {output_file}")
        print("⚠ If you want to download again, please delete the file first")
    return Path(output_file).exists()


@click.command()
@click.option("--file_url", help="Fits File containing star list")
@click.option("--destination", help="Destination file path")
def download(file_url, destination):
    download_file_all_star(file_url, destination)


if __name__ == "__main__":
    download()
