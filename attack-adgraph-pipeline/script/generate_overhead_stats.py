import os
import argparse


parser = argparse.ArgumentParser(description='Parse arguments (see description in source file)')
parser.add_argument('--eval-html-dir', type=str)
parser.add_argument('--dump-out', type=str, default="")
parser.add_argument('--out-fpath', type=str)
parser.add_argument('--dump', action="store_true")
args = parser.parse_args()

REVERSE_MAP = {
    "_cent": "_dist",
    "_dist": "_cent"
}


def read_eval_html_dir(eval_html_dir, delete_larger_files=False):
    per_domain_file_size_info = {}
    per_domain_file_info = {}
    fnames = os.listdir(eval_html_dir)
    
    for fname in fnames:
        if not fname.endswith(".html"):
            continue
        if fname.startswith("original_"):
            domain = fname.replace(".html", '').replace("original_", "")
            fsize = os.path.getsize(eval_html_dir + '/' + fname)
            per_domain_file_info[domain] = {
                "original_fname": fname,
                "perturbed_fnames": [],
            }
            per_domain_file_size_info[domain] = {
                "original_size": fsize,
                "perturbed_sizes": [],
            }
    for fname in fnames:
        if not fname.endswith(".html"):
            continue
        both_worked = False
        if not fname.startswith("original_"):
            domain, url_id = fname.replace(".html", '').split('_', 1)
            if len(url_id.split('_')) == 3:
                counter_fname = fname
                counter_fname = counter_fname.replace(counter_fname[-10:-5], REVERSE_MAP[fname[-10:-5]])
                if os.path.isfile(eval_html_dir + '/' + counter_fname):
                    both_worked = True
                    counter_fsize = os.path.getsize(eval_html_dir + '/' + counter_fname)
            fsize = os.path.getsize(eval_html_dir + '/' +  fname)
            if both_worked and fsize >= counter_fsize:
                cmd = "rm %s" % eval_html_dir + '/' + fname
                os.system(cmd)
                continue
            per_domain_file_size_info[domain]["perturbed_sizes"].append(fsize)
            per_domain_file_info[domain]["perturbed_fnames"].append(fname)
    return per_domain_file_size_info, per_domain_file_info


per_domain_file_size_info, per_domain_file_info = read_eval_html_dir(args.eval_html_dir)
overheads = []
all_plt_fnames = []

if args.dump:
    with open(args.dump_out, 'w') as fin:
        for domain, info in per_domain_file_info.items():
            fin.write(domain + '\n')
            for fname in info["perturbed_fnames"]:
                if len(fname.split('_')) == 4:
                    domain, _, url_id, strategy = fname.replace(".html", "").split("_")
                    fin.write(','.join([domain, url_id, strategy]) + '\n')

for domain, info in per_domain_file_size_info.items():
    original_size = info["original_size"]
    for fsize in info["perturbed_sizes"]:
        overheads.append((fsize - original_size) / original_size)

print("Mean size delta: %f" % (sum(overheads) / len(overheads)))
print("Median size delta: %f" % (sorted(overheads)[int(len(overheads) / 2)]))
