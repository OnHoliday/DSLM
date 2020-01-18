from os import pardir, listdir
from os.path import join, dirname

from pandas import read_csv

from data.io_plm import remove_extension, dataset_to_pickle


def dataset_to_pickle(base_folder, dataset_filename, output_file_path=None):
    
    dataset = read_csv(join(base_folder, dataset_filename), header=None)
    dataset_name = remove_extension(dataset_filename)

    if output_file_path == None:
        output_file_path = join('03_standardized')
    dataset_to_pickle(dataset, output_file_path, dataset_name)


def all_datasets_to_pickle():
    
    base_folder = join(dirname(__file__), pardir, 'datasets', '02_cleaned')

    for dataset_filename in listdir(base_folder):
        dataset_to_pickle(base_folder, dataset_filename)


if __name__ == '__main__':

    # all_datasets_to_pickle()
    
    base_folder = join(dirname(__file__), pardir, 'datasets', '02_cleaned')
    #===========================================================================
    # dataset_to_pickle(base_folder, 'c_android.csv')
    #===========================================================================
    
    """
    196 features
    1290 instances
    1 output (regression)
    """
    dataset_to_pickle(base_folder, 'r_kazuhiro.csv')
