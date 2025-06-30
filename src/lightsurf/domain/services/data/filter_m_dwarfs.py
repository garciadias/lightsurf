from pathlib import Path

from astropy.io import fits

MODULE_PATH = Path(__file__).parents[5]
ALL_STAR_PATH = f"{MODULE_PATH}/data/allStar-dr17-synspec_rev1.fits"


def filter_m_dwarfs(all_star):
    max_teff = 4100
    min_logg = 4
    print(f"⭐ getting stars with Teff <= {max_teff} and logg >= {min_logg}")
    teff_filter = all_star[1].data["TEFF"] <= 4100
    logg_filter = all_star[1].data["LOGG"] >= 4
    return all_star[1].data[teff_filter & logg_filter]


def save_fits_file(fits_table, output_path):
    fits.writeto(output_path, fits_table, overwrite=True)
    return Path(output_path).exists()


if __name__ == "__main__":
    output_path = f"{MODULE_PATH}/data/mdwarfs_DR17.fits"
    print("📤 Opening allStar file")
    all_star = fits.open(ALL_STAR_PATH)
    print("🔍 Filtering M dwarfs")
    m_dwarfs = filter_m_dwarfs(all_star)
    print(f"💾 Saving M dwarfs at {output_path}")
    saved = save_fits_file(m_dwarfs, output_path)
    if saved:
        print(f"🎉 M dwarfs saved at {output_path}")
    else:
        print(f"🚨 Error saving M dwarfs, try again or see file {__file__}")
