import argparse


parser = argparse.ArgumentParser(description='Parse arguments (see description in source file)')
parser.add_argument('--plt-log-fpath', type=str)
args = parser.parse_args()

with open(args.plt_log_fpath, 'r') as fin:
    per_domain_plt_info = {}
    data = fin.readlines()
    for row in data:
        row = row.strip()
        if len(row.split(',')) == 3 and row.split(',')[1] == "original":
            domain, _, plt = row.split(',')
            per_domain_plt_info[domain] = {
                "original_plt": int(float(plt)),
                "perturbed_plt": []
            }
    for row in data:
        row = row.strip()
        if len(row.split(',')) == 4:
            domain, url_id, strategy, plt = row.split(',')
            per_domain_plt_info[domain]["perturbed_plt"].append(int(float(plt)))
        if len(row.split(',')) == 3 and row.split(',')[1] != "original":
            domain, url_id, plt = row.split(',')
            per_domain_plt_info[domain]["perturbed_plt"].append(int(float(plt)))


plt_delta = []

for domain, info in per_domain_plt_info.items():
    original_plt = info["original_plt"]
    for plt in info["perturbed_plt"]:
        if original_plt == 0:
            continue
        plt_delta.append((plt - original_plt) / original_plt)

print("Mean PLT delta: %f" % (sum(plt_delta) / len(plt_delta)))
print("Median PLT delta: %f" % (sorted(plt_delta)[int(len(plt_delta) / 2)]))
