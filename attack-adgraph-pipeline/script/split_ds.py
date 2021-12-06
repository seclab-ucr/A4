import random
import argparse

DATA_DIR = '../data/'


def replace_letter_label_with_number(row):
    label = row.split(',')[-1]
    if 'NONAD' in label:
        label = label.replace('NONAD', '0')
    else:
        label = label.replace('AD', '1')
    fields = row.split(',')
    fields[-1] = label

    return ','.join(fields)


def read_ds(ds_fpath):
    with open(ds_fpath, 'r') as fin:
        data = fin.readlines()

    header = data[0]
    del data[0]

    for i in range(len(data)):
        row = data[i]
        replaced = replace_letter_label_with_number(row)
        data[i] = replaced

    return header, data


def randomly_sample_from_ds(ds, sample_size, only_ad=True):
    random.shuffle(ds)
    if only_ad:
        ad_cnt = 0
        ad, nonad = [], []
        for i in range(len(ds)):
            if ds[i].strip().split(',')[-1] == "1" and ad_cnt < sample_size:
                ad_cnt += 1
                ad.append(ds[i])
            else:
                nonad.append(ds[i])
        return ad, nonad
    else:
        return ds[:sample_size], ds[sample_size:]


def dump_ds(ds, header, ds_fpath):
    ds_dict = {}
    for row in ds:
        domain = row.split(',')[0]
        if domain not in ds_dict:
            ds_dict[domain] = [row]
        else:
            ds_dict[domain].append(row)

    ds_list = []
    with open(ds_fpath, 'w') as fout:
        fout.write(header)
        for domain, records in sorted(ds_dict.items(), key=lambda item: item[0]):
            records = sorted(records, key=lambda item: int(
                item.split(',')[1].split('_')[-1]))
            ds_list.extend(records)
        fout.writelines(ds_list)


parser = argparse.ArgumentParser(
    description='Parse arguments (see description in source file)')
parser.add_argument('--ds-fpath', default='dataset_1111.csv', type=str)
parser.add_argument('--target-ds-fpath',
                    default='target_dataset.csv', type=str)
parser.add_argument('--training-ds-fpath',
                    default='training_dataset.csv', type=str)
parser.add_argument('--sample-size', default=5000, type=int)
args = parser.parse_args()

header, ds = read_ds(DATA_DIR + args.ds_fpath)
target, training = randomly_sample_from_ds(ds, args.sample_size)
dump_ds(target, header, DATA_DIR + args.target_ds_fpath)
dump_ds(training, header, DATA_DIR + args.training_ds_fpath)
