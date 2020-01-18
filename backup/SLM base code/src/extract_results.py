
import os

from data.io_plm import get_benchmark_folder
from extract_to_csv import process_selected_benchmarks as process_to_csv
from extract_to_generalization_boxplots import process_all as process_to_boxplot
from extract_to_latex_tables import process_all as process_to_latex


def process_inner(benchmarks_paths):
    
    # process_to_csv()
    process_to_csv(benchmarks_paths)
    
    process_to_latex()
    
    process_to_boxplot()


def process_selected_benchmarks(benchmarks_paths):
    
    process_inner(benchmarks_paths)



def process_all():
    
    benchmarks_paths = []
    for folder in os.listdir(get_benchmark_folder()):
        path = os.path.join(get_benchmark_folder(), folder)
        for file in os.listdir(path):
            benchmarks_paths.append(os.path.join(get_benchmark_folder(), folder, file))
    
    process_inner(benchmarks_paths)


if __name__ == '__main__':
    
    # process_all()
    
    benchmarks_paths = []
    
    # ensembles
    benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_cancer', 'c_cancer_slm__2019_02_06__06_33_12.pkl')]
    benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_slm__2019_02_06__17_58_00.pkl')]
    benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_slm__2019_02_06__17_58_00.pkl')]
    benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_sonar', 'c_sonar_slm__2019_02_06__17_58_00.pkl')]
    
    #===========================================================================
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_concrete', 'r_concrete_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_concrete', 'r_concrete_mlp-sgd-adam__2019_02_05__21_14_31.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_concrete', 'r_concrete_mlp__2019_02_05__19_26_39.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_cancer', 'c_cancer_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_cancer', 'c_cancer_mlp-sgd-adam__2019_02_05__22_12_15.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_cancer', 'c_cancer_mlp__2019_02_05__19_12_26.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_sonar', 'c_sonar_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_sonar', 'c_sonar_mlp-sgd-adam__2019_02_05__22_12_15.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_sonar', 'c_sonar_mlp__2019_02_05__19_12_26.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_parkinsons', 'r_parkinsons_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_parkinsons', 'r_parkinsons_mlp-sgd-adam__2019_02_05__21_14_31.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_parkinsons', 'r_parkinsons_mlp__2019_02_05__18_43_45.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_music', 'r_music_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_music', 'r_music_mlp-sgd-adam__2019_02_05__21_14_31.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_music', 'r_music_mlp__2019_02_05__18_43_45.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_mlp-sgd-adam__2019_02_05__22_12_15.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_mlp__2019_02_05__18_10_54.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_mlp-sgd-adam__2019_02_05__22_12_15.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_mlp__2019_02_05__18_10_54.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_mlp-sgd-adam__2019_02_05__21_14_31.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_mlp__2019_02_05__06_13_51.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_mlp-sgd-adam__2019_02_05__21_14_31.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_mlp__2019_02_05__06_13_51.pkl')]
    # 
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_slm__2019_02_05__06_13_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_mlp-sgd-adam__2019_02_05__21_14_31.pkl')]
    # # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_mlp__2019_02_05__06_13_51.pkl')]  
    #===========================================================================
    
    # legacy
    #===========================================================================
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_slm__2019_01_28__16_57_48.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_mlp__2019_01_29__21_34_07.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_slm__2019_01_28__16_57_48.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_mlp__2019_01_30__13_55_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_slm__2019_01_28__16_57_48.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_mlp__2019_01_27__23_03_38.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_slm__2019_01_28__16_57_48.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_mlp__2019_01_27__23_03_38.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_slm__2019_01_28__16_57_48.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_mlp__2019_01_27__23_03_38.pkl')]
    #===========================================================================

    # legacy
    #===========================================================================
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_credit', 'c_credit_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'c_diabetes', 'c_diabetes_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_bio', 'r_bio_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_ppb', 'r_ppb_mlp__2019_01_27__00_08_44.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_slm__2019_01_27__00_08_20.pkl')]
    # benchmarks_paths += [os.path.join(get_benchmark_folder(), 'r_student', 'r_student_mlp__2019_01_27__00_08_44.pkl')]
    #===========================================================================
    
    process_selected_benchmarks(benchmarks_paths)
