import os
import argparse


HOME_DIR = os.getenv('HOME')


def read_original_domain_to_final_url_mapping(fpath):
    mapping = {}
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for row in data:
        row = row.strip()
        original_domain, final_url = row.split(',', 1)
        mapping[original_domain] = final_url
    return mapping


def read_eval_html_dir(eval_html_dir):
    eval_html_mapping = {}
    fnames = os.listdir(eval_html_dir)
    for fname in fnames:
        if fname.startswith("original_"):
            original_domain = fname.replace("original_", '').replace(".html", '')
            eval_html_mapping[original_domain] = []
    for fname in fnames:
        if not fname.startswith("original_"):
            if len(fname.replace(".html", '').split('_')) != 3:
                continue
            original_domain, url_id = fname.replace(".html", '').split('_', 1)
            eval_html_mapping[original_domain].append(url_id)
    return eval_html_mapping


parser = argparse.ArgumentParser(description='Parse arguments (see description in source file)')
parser.add_argument('--mapping-fpath', type=str)
parser.add_argument('--eval-html-dir', type=str)
args = parser.parse_args()

mapping = read_original_domain_to_final_url_mapping(args.mapping_fpath)
eval_html_mapping = read_eval_html_dir(args.eval_html_dir)

with open(HOME_DIR + "/rendering_stream/eval_mapping/eval_original_mapping.csv", 'w') as fout:
    for original_domain, final_url in mapping.items():
        if original_domain in eval_html_mapping:
            fout.write(','.join([original_domain, final_url]) + '\n')
        else:
            continue

for original_domain, final_url in mapping.items():
    if original_domain not in eval_html_mapping:
        continue
    url_ids = eval_html_mapping[original_domain]
    for url_id in url_ids:
        with open(HOME_DIR + "/rendering_stream/eval_mapping/eval_modified_mapping_%s_%s.csv" % (original_domain, url_id), 'w') as fout:
            fout.write(','.join([original_domain + '_' + url_id, final_url]) + '\n')
