import os
from shutil import copyfile
from numpy import floor

test_path = 'COVID-Net/data/test'
train_path = 'COVID-Net/data/train'

test_mapping_path = 'test_split_pneumonia.txt'
train_mapping_path = 'train_split_pneumonia.txt'

splitted_test_path = 'data/test'
splitted_train_path = 'data/train'

test_mapping_path = os.path.normpath(test_mapping_path)
train_mapping_path = os.path.normpath(train_mapping_path)

test_path = os.path.normpath(test_path)
train_path = os.path.normpath(train_path)

splitted_test_path = os.path.normpath(splitted_test_path)
splitted_train_path = os.path.normpath(splitted_train_path)

def prepare_folders(destination_path, mapping):
    for c in set( val for val in mapping.values()):
        try:
            os.mkdir(os.path.abspath(os.path.normpath(destination_path + f'/{c}')))
        except FileExistsError:
            pass

def prepare_map_dict(mapping_path):
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            splitted_line = line.split(' ')
            mapping[splitted_line[1]] = splitted_line[2]
    return mapping

def map_files_into_folders(source_path, destination_path, mapping):
    counter = 1
    skipped_counter = 1
    all_files_count = len(os.listdir(source_path))
    for image_name in os.listdir(source_path):
        if image_name in mapping:
            source_file_path = os.path.abspath(os.path.normpath(source_path + f'/{image_name}'))
            destination_file_path = os.path.abspath(os.path.normpath(destination_path + f'/{mapping[image_name]}/{image_name}'))
            copyfile(source_file_path, destination_file_path)
            print(f'[{int(floor(counter/all_files_count * 100))} %] copied from {source_file_path} ({mapping[image_name]}) to folder with path {destination_file_path}')
        else:
            print(f'No file named {image_name} in mapping')
            skipped_counter += 1
        counter += 1
    print(f'There was {skipped_counter - 1} [{int(floor(skipped_counter/all_files_count * 100))} %] skipped files that did not have mapping')

if __name__ == '__main__':
    mapping_test = prepare_map_dict(test_mapping_path)
    # prepare_folders(splitted_test_path, mapping_test)
    map_files_into_folders(test_path, splitted_test_path, mapping_test)
    # mapping_train = prepare_map_dict(train_mapping_path)
    # prepare_folders(splitted_train_path, mapping_train)
    # map_files_into_folders(train_path, splitted_train_path, mapping_train)