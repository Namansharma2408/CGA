import os
import urllib.request
import tarfile
import random

DATASETS = {
    "soc-LiveJournal1": "SNAP",
    "kron_g500-logn21": "DIMACS10",
    "hollywood-2009": "LAW",
    "soc-orkut": "SNAP",
    "indochina-2004": "LAW",

    "roadNet-CA": "SNAP",
    "delaunay_n24": "DIMACS10",
    "rgg_n_2_24_s0": "DIMACS10",
    "belgium_osm": "DIMACS10",
    "road_usa": "DIMACS10"
}

def download_and_extract():
    os.makedirs("data/realworld_graphs", exist_ok=True)
    os.makedirs("data/downloads", exist_ok=True)

    base_url = "https://sparse.tamu.edu/MM"

    print("Downloading exact real-world datasets from Table I (SuiteSparse Matrix Collection)...\n")

    dest_dir = "data/realworld_graphs"

    for name, group in DATASETS.items():
        url = f"{base_url}/{group}/{name}.tar.gz"
        tar_path = f"data/downloads/{name}.tar.gz"

        if not os.path.exists(f"{dest_dir}/{name}/{name}.mtx") and not os.path.exists(f"{dest_dir}/{name}.mtx"):
            print(f" -> Fetching {name}.mtx (Group: {group})")
            try:
                urllib.request.urlretrieve(url, tar_path)

                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=dest_dir)

                print(f"    [Success] Extracted to {dest_dir}/{name}/")

                os.remove(tar_path)
            except Exception as e:
                print(f"    [Failed] Could not download {name}: {e}")
        else:
            print(f" -> {name} already exists in {dest_dir}.")

    print("\nReal-World Dataset Acquisition Complete!")

if __name__ == "__main__":
    download_and_extract()