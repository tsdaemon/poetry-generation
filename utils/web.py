import os
import requests
from tqdm import tqdm


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    try:
        u = requests.get(url)
    except:
        print("URL %s failed to open" %url)
        raise Exception
    try:
        f = open(filepath, 'wb')
    except:
        print("Cannot write %s" %filepath)
        raise Exception

    downloaded = 0
    for buf in tqdm(u.iter_content(100000)):
        if not buf:
            break
        downloaded += len(buf)
        f.write(buf)
    f.close()
    return filepath