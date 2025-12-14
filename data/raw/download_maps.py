import csv
import os
import urllib.request
import urllib.parse
import logging
from bs4 import BeautifulSoup
import re
from tqdm import tqdm # Loading bars
from http.client import IncompleteRead
import time
import socket
from urllib.parse import urlparse, parse_qs

socket.setdefaulttimeout(30) # Longer timeout

# Configure logging for Jupyter Notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Direct logs to the notebook's output
    ]
)
#helpers
def sanitize_filename(s):
    return "".join(c for c in s if c.isalnum() or c in (' ', '_', '-')).rstrip()


def extract_f_from_href(href: str):
    # href is like: "index.php?lang_id=1&...&f=54333_shade.tif&..."
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    return qs.get("f", [None])[0]

def find_file_link_for_ruudunumber(html_content: str, ruudunumber: str, allowed_ext=None):
    soup = BeautifulSoup(html_content, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "dl=1" not in href or "f=" not in href:
            continue

        fname = extract_f_from_href(href)
        if not fname:
            continue

        # Ensure it’s the tile we asked for
        if not fname.startswith(f"{ruudunumber}_"):
            continue

        # Optional filter by extension
        if allowed_ext is not None:
            if not any(fname.lower().endswith(ext.lower()) for ext in allowed_ext):
                continue

        file_link = f"https://geoportaal.maaamet.ee/?{href}"
        return fname, file_link

    return None, None


def download_with_retries(url, file_path, retries=3, delay=2):
    for attempt in range(retries):
        try:
            logging.debug(f"Attempt {attempt + 1} to download {url}")
            response = urllib.request.urlopen(url)
            with open(file_path, 'wb') as file:
                file.write(response.read())
            logging.debug(f"Download successful: {file_path}")
            return  # Exit after successful download
        except IncompleteRead as e:
            logging.debug(f"Incomplete read error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            logging.debug(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logging.debug(f"Failed to download {url} after {retries} attempts.")

# url searches for files
def get_tava_file_url(ruudunumber):
    base_url = 'https://geoportaal.maaamet.ee/index.php'
    params = {
        'lang_id': '1',
        'plugin_act': 'otsing',
        'kaardiruut': ruudunumber,
        'andmetyyp': 'lidar_laz_tava',
        'page_id': '614'
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    response = urllib.request.urlopen(url)
    html_content = response.read().decode('utf-8')
    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a', href=True)
    for link in links:
        if 'tava.laz' in link['href']:
            match = re.search(fr'{ruudunumber}_(\d+)_tava\.laz', link['href'])
            if match:
                file_name = match[0]
                file_link = f"https://geoportaal.maaamet.ee/?{link['href']}"
                logging.debug(f'match found, returning: {match[0]} and {file_link}')
                return file_name, file_link
            
def get_dtm_file_url(ruudunumber):
    base_url = "https://geoportaal.maaamet.ee/index.php"
    params = {
        "lang_id": "1",
        "plugin_act": "otsing",
        "kaardiruut": ruudunumber,
        "andmetyyp": "dem_1m_geotiff",
        "page_id": "614",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    html_content = urllib.request.urlopen(url).read().decode("utf-8")

    # If you want to be strict, keep only the exact file:
    fname, link = find_file_link_for_ruudunumber(html_content, ruudunumber, allowed_ext=[".tif"])
    if fname and fname.endswith("_dtm_1m.tif"):
        return fname, link
    return None, None


def get_reljeef_file_url(ruudunumber):
    base_url = "https://geoportaal.maaamet.ee/index.php"
    params = {
        "lang_id": "1",
        "plugin_act": "otsing",
        "kaardiruut": ruudunumber,
        "andmetyyp": "reljeefivarjutus_hall",
        "page_id": "614",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    html_content = urllib.request.urlopen(url).read().decode("utf-8")

    fname, link = find_file_link_for_ruudunumber(html_content, ruudunumber, allowed_ext=[".tif"])
    # Your example is 54333_shade.tif; enforce that if you want:
    if fname and fname.endswith("_shade.tif"):
        return fname, link
    return None, None
 

def get_orto_file_url(ruudunumber):
    base_url = "https://geoportaal.maaamet.ee/index.php"
    params = {
        "lang_id": "1",
        "plugin_act": "otsing",
        "kaardiruut": ruudunumber,
        "andmetyyp": "ortofoto_eesti_rgb",
        "page_id": "610",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    html_content = urllib.request.urlopen(url).read().decode("utf-8")

    return find_file_link_for_ruudunumber(
        html_content,
        ruudunumber,
        allowed_ext=[".zip"]
    )

#download and save the files

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created output directory: {path}")

def already_downloaded(output_dir: str, filename: str) -> bool:
    return os.path.exists(os.path.join(output_dir, filename))

def download_one(get_url_fn, ruudunumber: str, output_dir: str, sleep_s: float = 0.5):
    """
    Calls get_url_fn(ruudunumber) -> (filename, url).
    If found and not already downloaded, downloads into output_dir.
    Returns True if downloaded, False otherwise.
    """
    if not output_dir:
        return False  # dataset disabled

    ensure_dir(output_dir)

    fname, url = get_url_fn(ruudunumber)
    time.sleep(sleep_s)  # reduce query rate

    if not url or not fname:
        logging.debug(f"No file found for ruudunumber {ruudunumber} using {get_url_fn.__name__}")
        return False

    if already_downloaded(output_dir, fname):
        logging.debug(f"Already exists, skipping: {os.path.join(output_dir, fname)}")
        return False

    out_path = os.path.join(output_dir, fname)
    download_with_retries(url, out_path)
    logging.info(f"Downloaded: {out_path}")
    return True

def process_csv(
    input_csv,
    output_dirs,
    sleep_s=0.5,
):
    """
    output_dirs example:
    {
      "laz": "../data/lazFiles/",
      "dtm": "../data/dtmFiles/",
      "reljeef": "../data/reljeefFiles/",
      "orto": "../data/ortoFiles/",
    }

    Set any value to None to disable that dataset.
    """

    # Map dataset keys to (function, which-column)
    # Column choice:
    # - laz uses 1:2000 -> row[1]
    # - others use 1:10000 -> row[2]
    dataset_plan = {
        "laz":     (get_tava_file_url,     1),
        "dtm":     (get_dtm_file_url,      2),
        "reljeef": (get_reljeef_file_url,  2),
        "orto":    (get_orto_file_url,     2),
    }

    # Make dirs (only those enabled)
    for key, out_dir in output_dirs.items():
        if out_dir:
            ensure_dir(out_dir)

    total_downloaded = 0
    total_skipped_or_missing = 0

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # skip header

        for row_number, row in tqdm(enumerate(reader, start=2)):
            if not row or len(row) < 3:
                logging.debug(f"Skipping line {row_number}: not enough columns.")
                continue

            if row[0].strip() == "Lisad":
                break

            linnamagi_name = row[0].strip()
            ruut_2000_raw = (row[1] or "").strip()
            ruut_10000_raw = (row[2] or "").strip()

            # Parse both columns into lists (support "463636, 462636" etc.)
            ruut_2000_list = [x.strip() for x in ruut_2000_raw.split(",") if x.strip()]
            ruut_10000_list = [x.strip() for x in ruut_10000_raw.split(",") if x.strip()]

            # If both missing, nothing to do
            if not ruut_2000_list and not ruut_10000_list:
                logging.debug(f"Skipping line {row_number} ({linnamagi_name}): both ruudunumbers missing.")
                continue

            # For each dataset: pick the right list, download for each ruudunumber
            for dataset_key, (get_fn, col_idx) in dataset_plan.items():
                out_dir = output_dirs.get(dataset_key)

                # Dataset disabled or not configured
                if not out_dir:
                    continue

                ruut_list = ruut_2000_list if col_idx == 1 else ruut_10000_list
                if not ruut_list:
                    logging.debug(
                        f"Line {row_number} ({linnamagi_name}): no ruudunumber for {dataset_key} (column {col_idx})."
                    )
                    continue

                for ruudunumber in ruut_list:
                    try:
                        did = download_one(get_fn, ruudunumber, out_dir, sleep_s=sleep_s)
                        if did:
                            total_downloaded += 1
                        else:
                            total_skipped_or_missing += 1
                    except Exception as e:
                        logging.error(
                            f"Error downloading {dataset_key} for ruudunumber {ruudunumber} "
                            f"on line {row_number} ({linnamagi_name}): {e}"
                        )
                        total_skipped_or_missing += 1
                        continue

    logging.info(f"Total downloaded files: {total_downloaded}")
    logging.info(f"Total skipped/missing/errors: {total_skipped_or_missing}")
