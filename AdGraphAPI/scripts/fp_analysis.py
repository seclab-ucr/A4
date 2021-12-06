import os
from os import listdir
from os.path import isfile, join


def get_files(file_path):
    only_files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith('.csv')]
    only_files = [os.path.join(file_path, f) for f in only_files]
    return only_files

def get_files_path(file_path):
    only_files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith('.csv')]
    only_files = [os.path.join(file_path, f) for f in only_files]
    return only_files

def read_file_lines(file_to_read):
    with open(file_to_read, 'r') as myfile:
        return myfile.readlines()


def write_file(file_name, lines_to_write):
    f = open(file_name, 'w')    
    for line in lines_to_write:
        f.write(line)  


def get_associated_url(file_path, url_id):
    file_lines = read_file_lines(file_path)
    for line in file_lines:
        if line.strip() != '' and line.split(',',1)[0].strip() == url_id.strip():
            return line.split(',',1)[1].strip()    
    return ''


visualization_files_1 = '/mnt/drive/work/visualization_1/'
visualization_files_2 = '/mnt/drive/work/visualization_2/'
visualization_files_3 = '/mnt/drive/work/visualization_3/'
visualization_files_4 = '/mnt/drive/work/visualization_4/'
visualization_files_5 = '/mnt/drive/work/visualization_5/'
visualization_files_6 = '/mnt/drive/work/visualization_6/'
visualization_files_7 = '/mnt/drive/work/visualization_7/'
visualization_files_8 = '/mnt/drive/work/visualization_8/'
visualization_files_9 = '/mnt/drive/work/visualization_9/'
visualization_files_10 = '/mnt/drive/work/visualization_10/'
visualization_files_11 = '/mnt/drive/work/visualization_11/'
visualization_files_12 = '/mnt/drive/work/visualization_12/'


visualization_files = get_files(visualization_files_1) +\
 get_files(visualization_files_2) +\
 get_files(visualization_files_3) +\
 get_files(visualization_files_4) +\
 get_files(visualization_files_5) +\
 get_files(visualization_files_6) +\
 get_files(visualization_files_7) +\
 get_files(visualization_files_8) +\
 get_files(visualization_files_9) +\
 get_files(visualization_files_10) +\
 get_files(visualization_files_11) +\
 get_files(visualization_files_12)


visualization_files_trunc = []
for v_file in visualization_files:
    visualization_files_trunc.append(v_file.split('/')[5][:-4])    


results_file = '/mnt/drive/work/results/unique_features_data_without_katz.result'
fp_file = '/mnt/drive/work/results/false_positives_all_v2.csv'

results = read_file_lines(results_file)

tp = 0
fp = 0
tn = 0
fn = 0

fp_list = []
fp_urls = []
fp_domains = []
list_to_write = []

for line in results[5:]:
    if line.strip() == '':
        continue

    actual = line.split(',')[1].split(':')[1]
    predicted = line.split(',')[2].split(':')[1]

    if actual == 'NONAD' and predicted == 'NONAD':
        tn += 1
    elif actual == 'NONAD' and predicted == 'AD':
        fp += 1
        fp_list.append(line.split(',')[5] + ',' + line.split(',')[6])
        tmp = line.split(',')[5]
        
        if tmp.startswith('\''):
            tmp = tmp[1:]
        if tmp.endswith('\''):
            tmp = tmp[:-1]

        if tmp.startswith('http://'):
            tmp = tmp[7:]
        elif tmp.startswith('https://'):
            tmp = tmp[8:]

        tmp = tmp.split('/')[0]

        if tmp in visualization_files_trunc:
            file_addr = visualization_files[visualization_files_trunc.index(tmp)]
            associated_url = get_associated_url(file_addr, line.split(',')[6]) 
            
            list_to_write.append(file_addr + ',' + tmp + ',' + line.split(',')[18].strip() + ',' + associated_url + '\n')
            if associated_url not in fp_urls:
                fp_urls.append(associated_url)     
            if tmp not in fp_domains:
                fp_domains.append(tmp)
                # print tmp                     
            # print file_addr, " , " , associated_url
        else:
            print '---------------------------------'
            print line.split(',')[5], tmp, line.split(',')[6]
            print '---------------------------------'
        
    elif actual == 'AD' and predicted == 'AD':
        tp += 1
    elif actual == 'AD' and predicted == 'NONAD':
        fn += 1

precision = float(tp) / float(tp + fp)
recall = float(tp) / float(tp + fn)
accuracy = float(tp + tn) / float(tp + tn + fp + fn)

print tp, tn, fp, fn    
print precision, recall, accuracy
print len(fp_urls)
print len(fp_domains)
write_file(fp_file, list_to_write)