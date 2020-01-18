from os import pardir, listdir
from os.path import join, dirname

from benchmark.results_extractor import generate_test_boxplot


def process_all():
    
    dataset_names = []
    dataset_folders = []
    
    base_folder = join(dirname(__file__), pardir, 'datasets', '06_formatted')
    
    for dataset_name in listdir(base_folder):
        dataset_names.append(dataset_name)
        dataset_folder = join(base_folder, dataset_name)
        dataset_folders.append(dataset_folder)
        
        min_max_y = None
        """
        if dataset_name == 'c_credit':
            min_max_y = [0.255, 0.868]
        elif dataset_name == 'c_diabetes':
            min_max_y = [0.172, 0.674]
        elif dataset_name == 'r_bio':
            min_max_y = [0.052, 1.354]
            # maximum y => intra-mlp
            # min_max_y = [0.052, 7.866]
            # with MLP ensembles: [-4.05988482e+145  8.52575812e+146]
        elif dataset_name == 'r_ppb':
            min_max_y = [0.032, 1.269]
            #===================================================================
            # [-25400.42022747 533408.9388544 ]
            # [0.03472334 1.26841793]
            # [-8.42050754e+06  1.76830658e+08]
            # [0.03396192 1.10412891]
            # [0.03396192 1.10412891]
            # [-2.73158150e+32  5.73632116e+33]
            #===================================================================
        elif dataset_name == 'r_student':
            min_max_y = [0.011, 1.282]
            #===================================================================
            # [-0.0766323  1.7018316]
            # [0.01276999 1.34692627]
            # [-0.06259755  1.40803776]
            # [-1.05672483e-03  1.28146338e+00]
            # [-1.05672483e-03  1.28146338e+00]
            # [-1.19564033e+64  2.51084470e+65]
            #===================================================================
        else:
            min_max_y = None
        """
        
        algos_list = ['slm_single', 'mlp_single']
        labels_list = ['SLM', 'MLP']
        output_filename = 'slm-vs-mlp-no-ensembles'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_auroc', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y, ylabel='AUROC')
        
        #=======================================================================
        # # SLM ensembles excluding RIW and including SLM without ensemble
        # algos_list = ['slm_single', 'slm_simple_ensemble', 'slm_bagging_ensemble', 'slm_boosting_1', 'slm_boosting_2', 'slm_boosting_3', 'slm_boosting_4']
        # labels_list = ['Single NN', 'Simple', 'Bagging', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'slm-ensembles-excluding-riw-and-including-single-nn'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        # 
        # algos_list = ['slm_simple_ensemble', 'slm_bagging_ensemble', 'slm_boosting_1', 'slm_boosting_2', 'slm_boosting_3', 'slm_boosting_4']
        # labels_list = ['Simple', 'Bagging', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'slm-ensembles'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_auroc', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y, ylabel='AUROC')
        # 
        # algos_list = ['slm_single', 'slm_simple_ensemble', 'slm_bagging_ensemble', 'slm_boosting_1', 'slm_boosting_2', 'slm_boosting_3', 'slm_boosting_4']
        # labels_list = ['Single NN', 'Simple', 'Bagging', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'slm-ensembles-with-single-nn'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_auroc', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y, ylabel='AUROC')
        #=======================================================================
        
        #=======================================================================
        # algos_list = ['slm_fls_group', 'slm_ols_group', 'slm_fls_tie_edv_group', 'slm_ols_edv']
        # labels_list = ['FLS variants', 'OLS variants', 'FLS + TIE/EDV', 'OLS + EDV']
        # output_filename = 'intra-slm'
        # generate_test_boxplot(dataset_folder, algos_list, 'testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        #=======================================================================

        #=======================================================================
        # algos_list = ['mlp_adam', 'mlp_sgd']
        # labels_list = ['Adam', 'SGD']
        # # algos_list = ['mlp_lbfgs', 'mlp_adam', 'mlp_sgd']
        # # labels_list = ['BFGS', 'Adam', 'SGD']
        # output_filename = 'intra-mlp'
        # generate_test_boxplot(dataset_folder, algos_list, 'testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        #=======================================================================

        #=======================================================================
        # # SLM ensembles including RIW
        # algos_list = ['slm_simple_ensemble', 'slm_bagging_ensemble', 'slm_riw_ensemble', 'slm_boosting_1', 'slm_boosting_2', 'slm_boosting_3', 'slm_boosting_4']
        # labels_list = ['Simple', 'Bagging', 'RIW', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'slm-ensembles-including-riw'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        # 
        # # SLM ensembles excluding RIW
        # algos_list = ['slm_simple_ensemble', 'slm_bagging_ensemble', 'slm_boosting_1', 'slm_boosting_2', 'slm_boosting_3', 'slm_boosting_4']
        # labels_list = ['Simple', 'Bagging', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'slm-ensembles-excluding-riw'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        # 
        # # SLM ensembles including RIW and including SLM without ensemble
        # algos_list = ['slm_single', 'slm_simple_ensemble', 'slm_bagging_ensemble', 'slm_riw_ensemble', 'slm_boosting_1', 'slm_boosting_2', 'slm_boosting_3', 'slm_boosting_4']
        # labels_list = ['Single NN', 'Simple', 'Bagging', 'RIW', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'slm-ensembles-including-riw-and-including-single-nn'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        # 
        # # MLP ensembles
        # algos_list = ['mlp_simple_ensemble', 'mlp_bagging_ensemble', 'mlp_boosting_1', 'mlp_boosting_2', 'mlp_boosting_3', 'mlp_boosting_4']
        # labels_list = ['Simple', 'Bagging', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'mlp-ensembles'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        # 
        # # MLP ensembles including MLP without ensemble
        # algos_list = ['mlp_single', 'mlp_simple_ensemble', 'mlp_bagging_ensemble', 'mlp_boosting_1', 'mlp_boosting_2', 'mlp_boosting_3', 'mlp_boosting_4']
        # labels_list = ['Single NN', 'Simple', 'Bagging', 'Boosting-1', 'Boosting-2', 'Boosting-3', 'Boosting-4']
        # output_filename = 'mlp-ensembles-including-single-nn'
        # generate_test_boxplot(dataset_folder, algos_list, 'best_results_testing_value', labels_list=labels_list, output_filename=output_filename, min_max_y=min_max_y)
        #=======================================================================


if __name__ == '__main__':
    
    process_all()
