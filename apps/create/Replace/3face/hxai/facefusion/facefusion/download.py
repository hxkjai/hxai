import os
import subprocess
import platform
import ssl
import urllib.request
from typing import List
from functools import lru_cache
from tqdm import tqdm

import facefusion.globals
from facefusion import wording
from facefusion.filesystem import is_file

if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not is_file(download_file_path):
            with open(download_file_path, 'w') as file:
                file.write('This file is not downloaded.')


@lru_cache(maxsize = None)
def get_download_size(url : str) -> int:
	try:
		response = urllib.request.urlopen(url, timeout = 10)
		return int(response.getheader('Content-Length'))
	except (OSError, ValueError):
		return 0


def is_download_done(url : str, file_path : str) -> bool:
	if is_file(file_path):
		return get_download_size(url) == os.path.getsize(file_path)
	return False
