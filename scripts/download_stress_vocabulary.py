import os

from utils.web import download


if __name__ == "__main__":
    data_dir = './data'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    database_name = os.path.join(data_dir, 'accents.csv')
    db_url = 'https://github.com/tsdaemon/uk-accents/raw/master/data/accents.csv'
    if not os.path.exists(database_name):
        download(db_url, data_dir)