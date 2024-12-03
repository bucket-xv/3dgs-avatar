
with open('../gdown/gdown/download_folder.py', 'r') as f:
    code = f.read().replace('MAX_NUMBER_FILES = 50', 'MAX_NUMBER_FILES = 10000')