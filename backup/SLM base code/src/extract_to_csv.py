import os

from benchmark.formatter import format_benchmark, merge_best_results, \
    general_format_benchmark
from data.io_plm import get_benchmark_folder, read_pickle, get_formatted_folder


def process_inner(benchmarks_paths):
    
    """ this block of code formats the benchmark files into csv files """
    for benchmark_path in benchmarks_paths:
        print(benchmark_path)
        benchmark = read_pickle(benchmark_path)
        #=======================================================================
        # format_benchmark(benchmark)
        #=======================================================================
        general_format_benchmark(benchmark)
    
    """ merge best results """
    for folder in os.listdir(get_formatted_folder()):
        path = os.path.join(get_formatted_folder(), folder)
        print(path)
        #=======================================================================
        # merge_best_results(path)
        #=======================================================================


def process_selected_benchmarks(benchmarks_paths):
    
    process_inner(benchmarks_paths)


def process_all():
    
    benchmarks_paths = []
    for folder in os.listdir(get_benchmark_folder()):
        path = os.path.join(get_benchmark_folder(), folder)
        for file in os.listdir(path):
            if file.__contains__('neat'):
                pass
            else:
                benchmarks_paths.append(os.path.join(get_benchmark_folder(), folder, file))
    
    process_inner(benchmarks_paths)


if __name__ == '__main__':
    
    # path = 'C:\Users\IvoGoncalves\Google Drive\Workspaces\Python\SLM\datasets\05_benchmark\c_cancer_generalization\c_cancer_generalization_ftne__2019_04_30__00_46_07.pkl'
    #===========================================================================
    # path = 'C:\\Users\\IvoGoncalves\\Google Drive\\Workspaces\\Python\\SLM\\datasets\\05_benchmark\\c_cancer_generalization\\c_cancer_generalization_ftne__2019_04_30__00_46_07.pkl'
    # benchmark = read_pickle(path)
    # print()
    #===========================================================================
    
    process_all()
    
    #===========================================================================
    # benchmarks_paths = []
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_slm__2019_01_27__00_08_20.pkl')]
    # #benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_mlp__2019_01_27__00_08_44.pkl')]
    # process_selected_benchmarks(benchmarks_paths)
    #===========================================================================
