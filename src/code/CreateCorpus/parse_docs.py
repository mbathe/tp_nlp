import glob
import os
from tqdm import tqdm
import magic

from Parser import Parser

current_working_directory = os.getcwd()

LOG_FILE = current_working_directory+"\\logs\\parse.log"
OUT_FOLDER = current_working_directory+"\\..\\..\\docfile\\txts\\"
log_fp = open(LOG_FILE, "w")

p = Parser(log_file=log_fp)

# Create output directory if it does not exist
if not os.path.exists(OUT_FOLDER):
    # print(OUT_FOLDER + " did not exist, created.")
    os.makedirs(OUT_FOLDER)

all_files = [f for f in glob.glob(current_working_directory+"\\..\\..\\docfile\\docs\\*")]

for i in tqdm(range(len(all_files))):
    fname = all_files[i]
    ftype = magic.from_file(fname, mime=True)

    if ftype == "text/html" or ftype == "text/xml":
        # this is a html file
        p.parse_html(fname)
    elif ftype == "application/pdf":
        # this is a pdf file
        p.parse_pdf(fname)
    else:
        # print(f"ERR. NOT A RECOGNIZED FILETYPE: {fname}, {ftype}.", file=log_fp)

log_fp.close()
