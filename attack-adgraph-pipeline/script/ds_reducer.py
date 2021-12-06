with open("../misc/alexa_top_actual_list.csv") as fin:
    data = fin.readlines()

cnt = 0
top_domain_set = set()
for i in range(len(data)):
    rank, alexa_domain, actual_domain = data[i].strip().split(",")
    top_domain_set.add(actual_domain)

with open("../data/training_dataset.csv") as fin:
    unnorm_ds = fin.readlines()
header = unnorm_ds[0]
del unnorm_ds[0]

target_set = []

for i in range(len(unnorm_ds)):
    is_target = False
    for domain in list(top_domain_set):
        if domain in unnorm_ds[i].split(',')[0]:
            is_target = True
            break
    if is_target:
        target_set.append(unnorm_ds[i])

with open("../data/target_dataset.csv", 'w') as fout:
    fout.writelines([header] + target_set)
