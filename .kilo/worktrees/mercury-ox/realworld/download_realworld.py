import os
import urllib.request
import gzip
import tarfile
import shutil

DATASETS = {
    "soc-LiveJournal1": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
    "ljournal-2008":    "https://sparse.tamu.edu/MM/LAW/ljournal-2008.tar.gz",
    "hollywood-2009":   "https://sparse.tamu.edu/MM/DIMACS10/hollywood-2009.tar.gz",
    "soc-orkut":        "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
    "roadNet-CA":       "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "realworld")

def download_file(url, dest):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)

def extract_gz(src, dest):
    print(f"Extracting {src} ...")
    with gzip.open(src, 'rb') as f_in:
        with open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, url in DATASETS.items():
        ext = ".tar.gz" if url.endswith(".tar.gz") else ".txt.gz"
        archive_path = os.path.join(DATA_DIR, name + ext)

        if not os.path.exists(archive_path):
            download_file(url, archive_path)

        if url.endswith(".txt.gz"):
            txt_path = os.path.join(DATA_DIR, name + ".txt")
            if not os.path.exists(txt_path):
                extract_gz(archive_path, txt_path)
        elif url.endswith(".tar.gz"):
            if not os.path.exists(os.path.join(DATA_DIR, name, name + ".mtx")):
                print(f"Extracting {archive_path} ...")
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR)

    print(f"\n✓ Realworld datasets downloaded to {DATA_DIR}")
    print("TIP: For the massive graphs (web-edu, road_usa), manually download from SuiteSparse Matrix Collection.")

if __name__ == "__main__":
    main()