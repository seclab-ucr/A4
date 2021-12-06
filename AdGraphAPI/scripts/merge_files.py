import os
from os import listdir
from os.path import isfile, join

def get_files(file_path):
    only_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    only_files = [os.path.join(file_path, f) for f in only_files]
    return only_files

def get_folders(folder_path):
    only_folders = [f for f in listdir(folder_path) if not isfile(join(folder_path, f))]
    only_folders = [os.path.join(folder_path, f) for f in only_folders]
    return only_folders    

def remove_files(folder_path):
    all_crawled_folders = get_folders(folder_path)
    for folder in all_crawled_folders:
        all_files = get_files(folder)

        for file_item in all_files:
            if 'parsed_' in file_item:
                os.remove(file_item)

def read_file_lines(file_to_read):
    with open(file_to_read, 'r') as myfile:
        return myfile.readlines()


def write_file(file_name, lines_to_write):
    f = open(file_name, 'w')
    for line in lines_to_write:
        f.write(line)  

extracted_feature_files_1 = '/mnt/drive/work/features_1/'
extracted_feature_files_2 = '/mnt/drive/work/features_2/'
extracted_feature_files_3 = '/mnt/drive/work/features_3/'
extracted_feature_files_4 = '/mnt/drive/work/features_4/'
extracted_feature_files_5 = '/mnt/drive/work/features_5/'
extracted_feature_files_6 = '/mnt/drive/work/features_6/'
extracted_feature_files_7 = '/mnt/drive/work/features_7/'
extracted_feature_files_8 = '/mnt/drive/work/features_8/'
extracted_feature_files_9 = '/mnt/drive/work/features_9/'
extracted_feature_files_10 = '/mnt/drive/work/features_10/'
extracted_feature_files_11 = '/mnt/drive/work/features_11/'
extracted_feature_files_12 = '/mnt/drive/work/features_12/'


feature_data_files_entries = get_files(extracted_feature_files_1) +\
 get_files(extracted_feature_files_2) +\
 get_files(extracted_feature_files_3) +\
 get_files(extracted_feature_files_4) +\
 get_files(extracted_feature_files_5) +\
 get_files(extracted_feature_files_6) +\
 get_files(extracted_feature_files_7) +\
 get_files(extracted_feature_files_8) +\
 get_files(extracted_feature_files_9) +\
 get_files(extracted_feature_files_10) +\
 get_files(extracted_feature_files_11) +\
 get_files(extracted_feature_files_12)


# feature_folder = '/mnt/drive/work/f_temp/'
all_feature_file = '/mnt/drive/work/unique_features_data.csv'
# all_files = get_files(feature_folder)

rows = []
unique_only = []
non_unique = []
for single_file in feature_data_files_entries:
    if single_file.split('/')[5] not in unique_only:
        rows += read_file_lines(single_file)
        unique_only.append(single_file.split('/')[5])
    else:
        non_unique.append(single_file)



write_file(all_feature_file, rows)
print len(unique_only)
print len(non_unique)
