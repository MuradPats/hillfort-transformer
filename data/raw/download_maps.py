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

socket.setdefaulttimeout(30) # Longer timeout

# Configure logging for Jupyter Notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Direct logs to the notebook's output
    ]
)

def sanitize_filename(s):
    return "".join(c for c in s if c.isalnum() or c in (' ', '_', '-')).rstrip()

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

def process_csv(input_csv, output_dir='lazFiles'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    skipping_ruudunumbrid = [i.split('_')[0] for i in os.listdir(output_dir)]
    counter = 0
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip the header line
        for row_number, row in tqdm(enumerate(reader, start=2)):  # Start at 2 to account for header
            if len(row) < 2:
                logging.debug(f"Skipping line {row_number}: not enough columns.")
                continue

            if row[0] == "Lisad":
                break

            first_col = row[0].strip()
            second_col = row[1].strip()

            if not second_col:
                logging.debug(f"Skipping line {row_number}: Ruudunumber is missing.")
                continue

            # Split second_col into a list of ruudunumbers
            ruudunumber_list = [num.strip() for num in second_col.split(',') if num.strip()]
            if not ruudunumber_list:
                logging.debug(f"Skipping line {row_number}: No valid Ruudunumber found.")
                continue

            for ruudunumber in ruudunumber_list:
                if ruudunumber in skipping_ruudunumbrid:
                    logging.debug(f"Skipping ruudunumber {ruudunumber} as its file already exists")
                    counter += 1
                    continue

                # Construct the URL using ruudunumber
                logging.debug(f"Ruudunumber: {ruudunumber}")
                f_param, file_url = get_tava_file_url(ruudunumber)
                # Reducing URL query rate
                time.sleep(0.5)

                # Download the file
                try:
                    logging.debug(f"Downloading from {file_url}...")
                    temp_filepath = os.path.join(output_dir, f_param)
                    download_with_retries(file_url, temp_filepath)
                    logging.debug(f"Downloaded to {temp_filepath}")
                    counter += 1
                except Exception as e:
                    logging.error(f"Error downloading {file_url} on line {row_number}: {e}")
                    continue

            # Rename the file
            # prefix = sanitize_filename(first_col)
            # new_filename = f"{prefix}_{f_param}"
            # new_filepath = os.path.join(output_dir, new_filename)
            # os.rename(temp_filepath, new_filepath)
            # logging.info(f"Renamed to {new_filepath}")
            # counter += 1

    logging.info(f"Total files: {counter}")
