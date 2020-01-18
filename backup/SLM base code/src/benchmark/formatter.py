import os

from numpy import mean, std, sqrt

from algorithms.common.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion, TrainingImprovementEffectivenessCriterion
from data.io_plm import _get_path_to_data_dir
import numpy as np
import pandas as pd


def _metric_in_dict(metric, d):
    return metric in d[0].keys()


def _get_dictionaries_by_metric(results, metric):
    return {k: results[k] for k in results.keys() if _metric_in_dict(metric, results[k])}


def _get_values_from_dictionary(dictionary, metric):
    return [d[metric] for d in dictionary if d is not None and metric in d]


def _summarize_metric(metric, summarizer=mean):
    return [summarizer([m[i] for m in metric]) for i in range(len(metric[0]))]


def _format_static_table(results, metric):
    dictionaries = _get_dictionaries_by_metric(results, metric)
    values = {k: _get_values_from_dictionary(dictionaries[k], metric) for k in dictionaries.keys()}
    return pd.DataFrame.from_dict(values)


# -IG- not called
def _get_avg_value(results, dict_to_get, value_to_get):
    dictionaries = _get_dictionaries_by_metric(results, dict_to_get)
    values = {k: _get_values_from_dictionary(dictionaries[k], dict_to_get) for k in dictionaries.keys()}
    values_to_get = {k: _get_values_from_dictionary(values[k], value_to_get) for k in dictionaries.keys()}
    values_saved = {}
    if value_to_get == 'stopping_criterion':
        for key, value in values_to_get.items(): 
            if type(value[0]) == MaxGenerationsCriterion:
                nr_generations = [item.max_generation for item in value]
                values_saved[key] = mean(nr_generations)
    else: 
        for key, value in values_to_get.items(): 
            if type(value[0]) != str: 
                values_saved[key] = mean(value)
    return pd.DataFrame.from_dict(values_saved, orient='index')


def _format_configuration_table(results, value_to_get):
    """formats number generations, number of layers, learning step value, subset ratio"""
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    values = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_to_get = {k: _get_values_from_dictionary(values[k], value_to_get) for k in dictionaries.keys()}
    values_saved = {} 
    if value_to_get == 'stopping_criterion':
        values_gen = {k: _get_values_from_dictionary(results[k], 'topology') for k in dictionaries.keys()}  # using topology because it has one value for each gen
        for key, value in values_gen.items():
            nr_generations = [len(item) - 1 for item in value]
            values_saved[key] = nr_generations
    elif value_to_get == 'layers':
        return pd.DataFrame.from_dict(values_to_get)
    elif value_to_get == 'learning_step':
        for key, value in values_to_get.items():
            if type(value[0]) != str:
                values_saved[key] = value
    elif value_to_get == 'subset_ratio':
        for key, value in values_to_get.items():
            if value:
                subset_ratio = [item for item in value]
                values_saved[key] = subset_ratio
        df = pd.DataFrame.from_dict(values_saved, orient='index')
        df = df.fillna(0)
        return df.T
    else:
        print('\n\t\t\t[_format_configuration_table] Should not happen!')
    return pd.DataFrame.from_dict(values_saved)


def _format_rst_rwt_frequency(results):
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    best_configurations = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_saved = {}
    for key, value in best_configurations.items(): 
        if (key == 'slm_fls_group' or key == 'slm_ols_group'):
            nr_no_RST_RWT = 0 
            nr_RST = 0
            nr_RWT = 0
            for run in value:
                if (run['random_sampling_technique'] == False and run['random_weighting_technique'] == False): 
                    nr_no_RST_RWT += 1
                elif (run['random_sampling_technique'] == True and run['random_weighting_technique'] == False):
                    nr_RST += 1
                elif (run['random_sampling_technique'] == False and run['random_weighting_technique'] == True):
                    nr_RWT += 1
                else:
                    print('\n\t\t\t[_format_rst_rwt_frequency] Should not happen!')
            values = [nr_no_RST_RWT, nr_RST, nr_RWT]
            values_saved[key] = values
    return pd.DataFrame(values_saved, index=['No RST and No RWT Frequency', 'RST Frequency', 'RWT Frequency'])


def _format_tie_edv_frequency(results):
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    best_configurations = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_saved = {}
    for key, value in best_configurations.items(): 
        if (key == 'slm_fls_tie_edv_group'): 
            nr_TIE = 0 
            nr_EDV = 0
            for run in value: 
                if type(run['stopping_criterion']) == ErrorDeviationVariationCriterion: 
                    nr_EDV += 1
                elif type(run['stopping_criterion']) == TrainingImprovementEffectivenessCriterion:
                    nr_TIE += 1
                else:
                    print('\n\t\t\t[_format_tie_edv_frequency] Should not happen!')
            values = [nr_EDV, nr_TIE]
            values_saved[key] = values
    return pd.DataFrame(values_saved, index=['EDV Frequency', 'TIE Frequency'])


def _format_slm_best_overall_configuration_frequency(best_result):
    slm_fls_group_frequency = 0
    slm_ols_group_frequency = 0
    slm_fls_tie_edv_group_frequency = 0
    slm_ols_edv_frequency = 0
    values = {} 
    for run in best_result:
        if run['best_overall_key'] == 'slm_fls_group':
            slm_fls_group_frequency += 1
        elif run['best_overall_key'] == 'slm_ols_group':
            slm_ols_group_frequency += 1
        elif run['best_overall_key'] == 'slm_fls_tie_edv_group':
            slm_fls_tie_edv_group_frequency += 1
        elif run['best_overall_key'] == 'slm_ols_edv':
            slm_ols_edv_frequency += 1
        else:
            print('\n\t\t\t[_format_slm_best_overall_configuration_frequency] Should not happen!')
    values['slm_fls_group'] = slm_fls_group_frequency
    values['slm_ols_group'] = slm_ols_group_frequency
    values['slm_fls_tie_edv_group'] = slm_fls_tie_edv_group_frequency
    values['slm_ols_edv'] = slm_ols_edv_frequency
    df = pd.DataFrame.from_dict(values, orient='index')  # check this
    df = df.T
    return df


def _format_mlp_configuration_table(results, value_to_get, metric=None):
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    values = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_to_get = {k: _get_values_from_dictionary(values[k], value_to_get) for k in dictionaries.keys()}
    values_saved = {}
    if value_to_get == 'learning_rate_init':
        for key, value in values_to_get.items(): 
            if value:
                learning_rate_values = [item for item in value]
                values_saved[key] = learning_rate_values
        return pd.DataFrame(values_saved) 
    elif metric == 'number_layers': 
        for key, value in values_to_get.items(): 
            nr_layers = [len(item) for item in value] 
            values_saved[key] = nr_layers
        return pd.DataFrame(values_saved)
    elif metric == 'number_neurons':
        for key, value in values_to_get.items(): 
            nr_neurons = [sum(item) for item in value]
            values_saved[key] = nr_neurons
        return pd.DataFrame(values_saved)
    # else:
        # print('\n\t\t\t[_format_mlp_configuration_table] Should not happen!')
    return pd.DataFrame.from_dict(values_to_get)


def _format_mlp_activation_function_frequency(results):
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    best_configurations = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_saved = {}
    for key, value in best_configurations.items(): 
        nr_logistic = 0 
        nr_relu = 0
        nr_tanh = 0
        for run in value: 
            if run['activation'] == 'logistic': 
                nr_logistic += 1
            elif run['activation'] == 'relu':
                nr_relu += 1 
            elif run['activation'] == 'tanh':
                nr_tanh += 1
            else:
                print('\n\t\t\t[_format_mlp_activation_function_frequency] Should not happen!')
        values = [nr_logistic, nr_relu, nr_tanh]
        values_saved[key] = values
    return pd.DataFrame(values_saved, index=['Logistic Frequency', 'Relu Frequency', 'Tanh Frequency'])


def _format_mlp_penalty_frequency(results):
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    best_configurations = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_saved = {}
    for key, value in best_configurations.items(): 
        nr_penalty = 0 
        nr_no_penalty = 0
        for run in value: 
            if run['alpha'] == 0: 
                nr_no_penalty += 1
            elif run['alpha'] != 0:
                nr_penalty += 1
            else:
                print('\n\t\t\t[_format_mlp_penalty_frequency] Should not happen!')
        values = [nr_no_penalty, nr_penalty]
        values_saved[key] = values
    return pd.DataFrame(values_saved, index=['No Penalty Frequency', 'Penalty Frequency'])


def _format_mlp_best_overall_configuration_frequency(best_result, classification):
    lbfgs_frequency = 0
    adam_frequency = 0
    sgd_frequency = 0
    values = {} 
    for run in best_result:
        if run['best_overall_configuration']['solver'] == 'lbfgs':
            lbfgs_frequency += 1
        elif run['best_overall_configuration']['solver'] == 'adam':
            adam_frequency += 1
        elif run['best_overall_configuration']['solver'] == 'sgd':
            sgd_frequency += 1
        else:
            print('\n\t\t\t[_format_mlp_best_overall_configuration_frequency] Should not happen!')
    if classification: 
        # values['mlpc_lbfgs'] = lbfgs_frequency
        values['mlpc_adam'] = adam_frequency
        values['mlpc_sgd'] = sgd_frequency
    else: 
        # values['mlpr_lbfgs'] = lbfgs_frequency
        values['mlpr_adam'] = adam_frequency
        values['mlpr_sgd'] = sgd_frequency
    df = pd.DataFrame.from_dict(values, orient='index')  # check this
    df = df.T
    return df

#===============================================================================
# # with lbfgs
# def _format_mlp_best_overall_configuration_frequency(best_result, classification):
#     lbfgs_frequency = 0
#     adam_frequency = 0
#     sgd_frequency = 0
#     values = {} 
#     for run in best_result:
#         if run['best_overall_configuration']['solver'] == 'lbfgs':
#             lbfgs_frequency += 1
#         elif run['best_overall_configuration']['solver'] == 'adam':
#             adam_frequency += 1
#         elif run['best_overall_configuration']['solver'] == 'sgd':
#             sgd_frequency += 1
#         else:
#             print('\n\t\t\t[_format_mlp_best_overall_configuration_frequency] Should not happen!')
#     if classification: 
#         values['mlpc_lbfgs'] = lbfgs_frequency
#         values['mlpc_adam'] = adam_frequency
#         values['mlpc_sgd'] = sgd_frequency
#     else: 
#         values['mlpr_lbfgs'] = lbfgs_frequency
#         values['mlpr_adam'] = adam_frequency
#         values['mlpr_sgd'] = sgd_frequency
#     df = pd.DataFrame.from_dict(values, orient='index')  # check this
#     df = df.T
#     return df
#===============================================================================


def _format_mlp_sgd_adam_table(results, value_to_get):
    dictionaries = _get_dictionaries_by_metric(results, 'best_configuration')
    values = {k: _get_values_from_dictionary(dictionaries[k], 'best_configuration') for k in dictionaries.keys()}
    values_to_get = {k: _get_values_from_dictionary(values[k], value_to_get) for k in dictionaries.keys()}
    values_saved = {}
    for key, value in values_to_get.items(): 
        if value: 
            values_saved[key] = value
    return pd.DataFrame.from_dict(values_saved)


def _format_processing_time_table(results):
    dictionaries = _get_dictionaries_by_metric(results, 'processing_time')
    values = {k: _get_values_from_dictionary(
        dictionaries[k], 'processing_time') for k in dictionaries.keys()}
    for key, value in values.items():
        values[key] = [sum(item) for item in value]
    return pd.DataFrame.from_dict(values)


def _format_topology_table(results, component):
    dictionaries = _get_dictionaries_by_metric(results, 'topology')
    values = {k: _get_values_from_dictionary(dictionaries[k], 'topology') for k in dictionaries.keys()}
    values = {key: [item[-1] for item in value] for key, value in values.items()}
    values = {key: [item[component] for item in value] for key, value in values.items()}
    return pd.DataFrame.from_dict(values)


def _format_evo_table(results, metric):
    dictionaries = _get_dictionaries_by_metric(results, metric)
    values = {k: _get_values_from_dictionary(dictionaries[k], metric) for k in dictionaries.keys()}
    values = {key: [[item[i] for item in value if i < len(item)]
                    for i in range(max([len(item) for item in value]))] for key, value in values.items()}

    max_len = max(len(value) for key, value in values.items())

    mean_dict = {key: [mean(item) for item in value] for key, value in values.items()}

    se_dict = {key: [std(item) / sqrt(len(item)) for item in value]
               for key, value in values.items()}

    for key, value in mean_dict.items():
        delta_len = max_len - len(value)
        mean_dict[key].extend([np.nan for i in range(delta_len)])

    for key, value in se_dict.items():
        delta_len = max_len - len(value)
        se_dict[key].extend([np.nan for i in range(delta_len)])

    return pd.DataFrame.from_dict(mean_dict), pd.DataFrame.from_dict(se_dict)


def _format_static_list(best_results, metric, algo):
    values = {algo: [None for k in range(len(best_results))]}
    i = 0
    for run in best_results:
        values[algo][i] = run[metric]
        i += 1
    return pd.DataFrame.from_dict(values)


def format_results(results, classification):
    formatted_results = {}
    
    if classification:
        formatted_results['slm_training_accuracy'] = _format_static_table(results, 'training_accuracy')
        formatted_results['slm_testing_accuracy'] = _format_static_table(results, 'testing_accuracy')
    else:
        formatted_results['slm_training_value'] = _format_static_table(results, 'training_value')
        formatted_results['slm_testing_value'] = _format_static_table(results, 'testing_value')
    
    formatted_results['slm_best_configuration'] = _format_static_table(results, 'best_configuration')
    # TEMP
    #===========================================================================
    # formatted_results['slm_training_value'] = _format_static_table(results, 'training_value')
    # formatted_results['slm_testing_value'] = _format_static_table(results, 'testing_value')
    #===========================================================================
    # formatted_results['slm_processing_time'] = _format_processing_time_table(results)
    formatted_results['slm_avg_inner_training_error'] = _format_static_table(results, 'avg_inner_training_error')
    formatted_results['slm_avg_inner_validation_error'] = _format_static_table(results, 'avg_inner_validation_error')
    
    if classification:
        formatted_results['slm_avg_inner_training_accuracy'] = _format_static_table(results, 'avg_inner_training_accuracy')
        formatted_results['slm_avg_inner_validation_accuracy'] = _format_static_table(results, 'avg_inner_validation_accuracy')
    
    formatted_results['slm_number_generations'] = _format_configuration_table(results, 'stopping_criterion')
    formatted_results['slm_learning_step_value'] = _format_configuration_table(results, 'learning_step')
    formatted_results['slm_number_layers'] = _format_configuration_table(results, 'layers')
    
    #===========================================================================
    # formatted_results['slm_subset_ratio'] = _format_configuration_table(results, 'subset_ratio')
    # formatted_results['slm_RST_RWT_frequency'] = _format_rst_rwt_frequency(results)
    # formatted_results['slm_TIE_EDV_frequency'] = _format_tie_edv_frequency(results)
    #===========================================================================
    
    formatted_results['slm_training_time'] = _format_static_table(results, 'training_time')
    
    # formatted_results['number_neurons'] = _format_topology_table(results, 'neurons')
    # formatted_results['number_connections'] = _format_topology_table(results, 'connections')
    # formatted_results['training_value_evolution'] = _format_evo_table(
    #    results, 'training_value_evolution')
    # formatted_results['testing_value_evolution'] = _format_evo_table(
    #    results, 'testing_value_evolution')
    # formatted_results['processing_time_evolution'] = _format_evo_table(results, 'processing_time')
    
    return formatted_results


def format_results_mlp(results, classification):
    formatted_results = {}
    
    if classification:
        formatted_results['mlp_training_accuracy'] = _format_static_table(results, 'training_accuracy')
        formatted_results['mlp_testing_accuracy'] = _format_static_table(results, 'testing_accuracy')
    else:
        formatted_results['mlp_training_value'] = _format_static_table(results, 'training_value')
        formatted_results['mlp_testing_value'] = _format_static_table(results, 'testing_value')
    
    formatted_results['mlp_best_configuration'] = _format_static_table(results, 'best_configuration')
    # TEMP
    #===========================================================================
    # formatted_results['mlp_training_value'] = _format_static_table(results, 'training_value')
    # formatted_results['mlp_testing_value'] = _format_static_table(results, 'testing_value')
    #===========================================================================
    formatted_results['mlp_avg_inner_training_error'] = _format_static_table(results, 'avg_inner_training_error')
    formatted_results['mlp_avg_inner_validation_error'] = _format_static_table(results, 'avg_inner_validation_error')
    
    if classification:
        formatted_results['mlp_avg_inner_training_accuracy'] = _format_static_table(results, 'avg_inner_training_accuracy')
        formatted_results['mlp_avg_inner_validation_accuracy'] = _format_static_table(results, 'avg_inner_validation_accuracy')
    
    formatted_results['mlp_number_iterations'] = _format_mlp_configuration_table(results, 'max_iter')
    formatted_results['mlp_learning_rate'] = _format_mlp_configuration_table(results, 'learning_rate_init')
    formatted_results['mlp_number_layers'] = _format_mlp_configuration_table(results, 'hidden_layer_sizes', 'number_layers')
    formatted_results['mlp_number_neurons'] = _format_mlp_configuration_table(results, 'hidden_layer_sizes', 'number_neurons')  # totals, considering all the hidden layers 
    formatted_results['mlp_alpha'] = _format_mlp_configuration_table(results, 'alpha')
    formatted_results['mlp_activation_function_frequency'] = _format_mlp_activation_function_frequency(results)
    formatted_results['mlp_penalty_frequency'] = _format_mlp_penalty_frequency(results)
    formatted_results['mlp_batch_size'] = _format_mlp_sgd_adam_table(results, 'batch_size')
    formatted_results['mlp_shuffle'] = _format_mlp_sgd_adam_table(results, 'shuffle')
    formatted_results['mlp_momentum'] = _format_mlp_sgd_adam_table(results, 'momentum')
    formatted_results['mlp_nesterovs_momentum'] = _format_mlp_sgd_adam_table(results, 'nesterovs_momentum')
    formatted_results['mlp_beta_1'] = _format_mlp_sgd_adam_table(results, 'beta_1')
    formatted_results['mlp_beta_2'] = _format_mlp_sgd_adam_table(results, 'beta_2')
    # formatted_results['mlp_training_time'] = _format_static_table(results, 'training_time')
    # formatted_results['mlp_best_overall_configuration_frequency'] = _format_mlp_best_overall_configuration_frequency(results)
    return formatted_results 


def format_ensemble_results(formatted_benchmark, ensemble_results, classification, algo):
    
    if classification: 
        formatted_benchmark[algo + '_ensemble_training_accuracy'] = _format_static_table(ensemble_results, 'training_accuracy')
        formatted_benchmark[algo + '_ensemble_testing_accuracy'] = _format_static_table(ensemble_results, 'testing_accuracy')
    
    formatted_benchmark[algo + '_ensemble_training_value'] = _format_static_table(ensemble_results, 'training_value')
    formatted_benchmark[algo + '_ensemble_testing_value'] = _format_static_table(ensemble_results, 'testing_value')
    formatted_benchmark[algo + '_ensemble_base_algorithm'] = _format_static_table(ensemble_results, 'algorithm')
    # formatted_benchmark[algo + '_ensemble_training_time'] = _format_static_table(ensemble_results, 'training_time')
    return formatted_benchmark 


def format_best_result(formatted_benchmark, best_result, classification, algo):
    
    if classification:
        formatted_benchmark[algo + '_best_result_training_accuracy'] = _format_static_list(best_result, 'training_accuracy', algo)
        formatted_benchmark[algo + '_best_result_testing_accuracy'] = _format_static_list(best_result, 'testing_accuracy', algo)
        
        formatted_benchmark[algo + '_best_result_training_value'] = formatted_benchmark[algo + '_best_result_training_accuracy']
        formatted_benchmark[algo + '_best_result_testing_value'] = formatted_benchmark[algo + '_best_result_testing_accuracy']
    
    else:
        formatted_benchmark[algo + '_best_result_training_value'] = _format_static_list(best_result, 'training_value', algo)
        formatted_benchmark[algo + '_best_result_testing_value'] = _format_static_list(best_result, 'testing_value', algo)
    
    # TEMP
    #===========================================================================
    # formatted_benchmark[algo + '_best_result_training_value'] = _format_static_list(best_result, 'training_value', algo)
    # formatted_benchmark[algo + '_best_result_testing_value'] = _format_static_list(best_result, 'testing_value', algo)
    #===========================================================================
    formatted_benchmark[algo + '_best_result_configuration'] = _format_static_list(best_result, 'best_overall_configuration', algo)
    # formatted_benchmark[algo + '_best_result_processing_time'] = _format_static_list(best_result, 'processing_time', algo)
    # formatted_benchmark[algo + '_best_result_training_time'] = _format_static_list(best_result, 'training_time', algo)
    
    #===========================================================================
    # if algo == 'slm':
    #     formatted_benchmark['slm_best_overall_configuration_frequency'] = _format_slm_best_overall_configuration_frequency(best_result)
    # elif algo == 'mlp':
    #     formatted_benchmark['mlp_best_overall_configuration_frequency'] = _format_mlp_best_overall_configuration_frequency(best_result, classification)
    # else:
    #     print('\n\t\t\t[format_best_result] Should not happen!')
    #===========================================================================
    
    return formatted_benchmark

#===============================================================================
# def merge_best_results_testing_value(formatted_benchmark):
# 
#     def merge(dict1, dict2): return(dict2.update(dict1)) 
# 
#     slm_testing_value = formatted_benchmark['slm_best_result_testing_value']
#     mlp_testing_value = formatted_benchmark['mlp_best_result_testing_value']
#     
#     # TEMP
#     #===========================================================================
#     # slm_ensemble_testing_value = formatted_benchmark['slm_ensemble_testing_value']
#     #===========================================================================
#     
#     # TEMP
#     #===========================================================================
#     # mlp_ensemble_testing_value = formatted_benchmark['mlp_ensemble_testing_value']
#     #===========================================================================
#     
#     values = {}
#     values['slm_testing_value'] = slm_testing_value
#     values['mlp_testing_value'] = mlp_testing_value
#     
#     # TEMP
#     #===========================================================================
#     # values = merge(values, slm_ensemble_testing_value)
#     #===========================================================================
#     
#     # TEMP
#     #===========================================================================
#     # values = merge(values, mlp_ensemble_testing_value)
#     #===========================================================================
#===============================================================================


def relabel_model_names(model_names, model_names_dict, short=True):
    key = 'name_short' if short else 'name_long'
    return [model_names_dict[model_name][key] for model_name in model_names]


def relabel_ensemble_model_names(model_names, ensemble_names_dict, algo, short=True):
    key = 'name_short' if short else 'name_long'
    return [algo + ' ' + ensemble_names_dict[model_name][key] for model_name in model_names]


def relabel_best_model_names(model_names, algo):
    # TEMP
    #return [algo.upper() for model_name in model_names]
    return [model_name.upper() for model_name in model_names]

def general_format_results(formatted_results, results, classification, algo):
    
    #print(results[algo][0])
    entries = results[algo][0]
    
    if classification:
        formatted_results[algo + '_training_accuracy'] = _format_static_table(results, 'training_accuracy')
        formatted_results[algo + '_testing_accuracy'] = _format_static_table(results, 'testing_accuracy')
    else:
        formatted_results[algo + '_training_value'] = _format_static_table(results, 'training_value')
        formatted_results[algo + '_testing_value'] = _format_static_table(results, 'testing_value')
    
    formatted_results[algo + '_best_configuration'] = _format_static_table(results, 'best_configuration')
    # TEMP
    #===========================================================================
    # formatted_results[algo + '_training_value'] = _format_static_table(results, 'training_value')
    # formatted_results[algo + '_testing_value'] = _format_static_table(results, 'testing_value')
    #===========================================================================
    # formatted_results[algo + '_processing_time'] = _format_processing_time_table(results)
    formatted_results[algo + '_avg_inner_training_error'] = _format_static_table(results, 'avg_inner_training_error')
    formatted_results[algo + '_avg_inner_validation_error'] = _format_static_table(results, 'avg_inner_validation_error')
    
    if classification:
        formatted_results[algo + '_avg_inner_training_accuracy'] = _format_static_table(results, 'avg_inner_training_accuracy')
        formatted_results[algo + '_avg_inner_validation_accuracy'] = _format_static_table(results, 'avg_inner_validation_accuracy')
    
    formatted_results[algo + '_number_generations'] = _format_configuration_table(results, 'stopping_criterion')
    if 'learning_step' in entries:
        formatted_results[algo + '_learning_step_value'] = _format_configuration_table(results, 'learning_step')
    if 'layers' in entries:
        formatted_results[algo + '_number_layers'] = _format_configuration_table(results, 'layers')
    
    #===========================================================================
    # formatted_results[algo + '_subset_ratio'] = _format_configuration_table(results, 'subset_ratio')
    # formatted_results[algo + '_RST_RWT_frequency'] = _format_rst_rwt_frequency(results)
    # formatted_results[algo + '_TIE_EDV_frequency'] = _format_tie_edv_frequency(results)
    #===========================================================================
    
    formatted_results[algo + '_training_time'] = _format_static_table(results, 'training_time')
    
    # formatted_results['number_neurons'] = _format_topology_table(results, 'neurons')
    # formatted_results['number_connections'] = _format_topology_table(results, 'connections')
    # formatted_results['training_value_evolution'] = _format_evo_table(
    #    results, 'training_value_evolution')
    # formatted_results['testing_value_evolution'] = _format_evo_table(
    #    results, 'testing_value_evolution')
    # formatted_results['processing_time_evolution'] = _format_evo_table(results, 'processing_time')
    
    """ MLP part """
    formatted_results[algo + '_number_iterations'] = _format_mlp_configuration_table(results, 'max_iter')
    formatted_results[algo + '_learning_rate'] = _format_mlp_configuration_table(results, 'learning_rate_init')
    formatted_results[algo + '_number_layers'] = _format_mlp_configuration_table(results, 'hidden_layer_sizes', 'number_layers')
    formatted_results[algo + '_number_neurons'] = _format_mlp_configuration_table(results, 'hidden_layer_sizes', 'number_neurons')  # totals, considering all the hidden layers 
    formatted_results[algo + '_alpha'] = _format_mlp_configuration_table(results, 'alpha')
    #formatted_results[algo + '_activation_function_frequency'] = _format_mlp_activation_function_frequency(results)
    #formatted_results[algo + '_penalty_frequency'] = _format_mlp_penalty_frequency(results)
    formatted_results[algo + '_batch_size'] = _format_mlp_sgd_adam_table(results, 'batch_size')
    formatted_results[algo + '_shuffle'] = _format_mlp_sgd_adam_table(results, 'shuffle')
    formatted_results[algo + '_momentum'] = _format_mlp_sgd_adam_table(results, 'momentum')
    formatted_results[algo + '_nesterovs_momentum'] = _format_mlp_sgd_adam_table(results, 'nesterovs_momentum')
    formatted_results[algo + '_beta_1'] = _format_mlp_sgd_adam_table(results, 'beta_1')
    formatted_results[algo + '_beta_2'] = _format_mlp_sgd_adam_table(results, 'beta_2')
    # formatted_results[algo + '_training_time'] = _format_static_table(results, 'training_time')
    # formatted_results[algo + '_best_overall_configuration_frequency'] = _format_mlp_best_overall_configuration_frequency(results)
    
    return formatted_results

def general_format_best_result(formatted_benchmark, best_result, classification, algo):
    
    if classification:
        formatted_benchmark[algo + '_best_result_training_accuracy'] = _format_static_list(best_result, 'training_accuracy', algo)
        formatted_benchmark[algo + '_best_result_testing_accuracy'] = _format_static_list(best_result, 'testing_accuracy', algo)
        
        formatted_benchmark[algo + '_best_result_training_value'] = formatted_benchmark[algo + '_best_result_training_accuracy']
        formatted_benchmark[algo + '_best_result_testing_value'] = formatted_benchmark[algo + '_best_result_testing_accuracy']
    
    else:
        formatted_benchmark[algo + '_best_result_training_value'] = _format_static_list(best_result, 'training_value', algo)
        formatted_benchmark[algo + '_best_result_testing_value'] = _format_static_list(best_result, 'testing_value', algo)
    
    # TEMP
    #===========================================================================
    # formatted_benchmark[algo + '_best_result_training_value'] = _format_static_list(best_result, 'training_value', algo)
    # formatted_benchmark[algo + '_best_result_testing_value'] = _format_static_list(best_result, 'testing_value', algo)
    #===========================================================================
    formatted_benchmark[algo + '_best_result_configuration'] = _format_static_list(best_result, 'best_overall_configuration', algo)
    # formatted_benchmark[algo + '_best_result_processing_time'] = _format_static_list(best_result, 'processing_time', algo)
    # formatted_benchmark[algo + '_best_result_training_time'] = _format_static_list(best_result, 'training_time', algo)
    
    #===========================================================================
    # if algo == 'slm':
    #     formatted_benchmark['slm_best_overall_configuration_frequency'] = _format_slm_best_overall_configuration_frequency(best_result)
    # elif algo == 'mlp':
    #     formatted_benchmark['mlp_best_overall_configuration_frequency'] = _format_mlp_best_overall_configuration_frequency(best_result, classification)
    # else:
    #     print('\n\t\t\t[format_best_result] Should not happen!')
    #===========================================================================
    
    return formatted_benchmark

def general_format_benchmark(benchmark):
    
    output_path = os.path.join(_get_path_to_data_dir(), '06_formatted', benchmark.dataset_name)
    
    # If 'file_path_ext' does not exist, create 'file_path_ext'.
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if _is_classification(benchmark):
        if 'mlpr' in benchmark.results.keys():
            del benchmark.results['mlpr']
        if 'rfr' in benchmark.results.keys():
            del benchmark.results['rfr']
    
    formatted_benchmark = {}
    for algo in benchmark.models.keys():
        formatted_benchmark = general_format_results(formatted_benchmark, benchmark.results, benchmark.classification, algo)
        formatted_benchmark = general_format_best_result(formatted_benchmark, benchmark.best_result, benchmark.classification, algo)
    
    model_names_dict = get_model_names_dict(benchmark)
    
    #ensemble_names_dict = get_ensemble_names_dict(benchmark)
    # TEMP
    algo = None
    
    for key, value in formatted_benchmark.items():
        if 'evolution' in key:
            i = 0
            for tbl in value:
                if i == 0:
                    ext = 'mean'
                else:
                    ext = 'se'
                tbl.columns = relabel_model_names(tbl.columns, model_names_dict)
                path = os.path.join(output_path, key + '_' + ext + '.csv')
                tbl.to_csv(path)
                i += 1
        elif 'ensemble' in key:
            if 'slm' in key: 
                algo = 'SLM'
            elif 'mlp' in key: 
                algo = 'MLP'
            else:
                print('\n\t\t\t[format_benchmark, 2 of 2] Should not happen!')
            
            formatted_benchmark[key].columns = relabel_ensemble_model_names(value.columns, ensemble_names_dict, algo)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)
        elif 'best_result' in key:
            # TEMP
            formatted_benchmark[key].columns = relabel_best_model_names(value.columns, algo)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)
        else: 
            formatted_benchmark[key].columns = relabel_model_names(value.columns, model_names_dict)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)
    


def format_benchmark(benchmark):
    
    output_path = os.path.join(_get_path_to_data_dir(), '06_formatted', benchmark.dataset_name)
    
    # If 'file_path_ext' does not exist, create 'file_path_ext'.
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if _is_classification(benchmark):
        if 'mlpr' in benchmark.results.keys():
            del benchmark.results['mlpr']
        if 'rfr' in benchmark.results.keys():
            del benchmark.results['rfr']
    
    if 'slm_fls_group' in benchmark.models.keys():
        formatted_benchmark = format_results(benchmark.results, benchmark.classification)
        # TEMP
        #=======================================================================
        # formatted_benchmark = format_ensemble_results(formatted_benchmark, benchmark.results_ensemble, benchmark.classification, 'slm')
        #=======================================================================
        formatted_benchmark = format_best_result(formatted_benchmark, benchmark.best_result, benchmark.classification, 'slm')
    elif ('mlpc_sgd' in benchmark.models.keys() or 'mlpr_sgd' in benchmark.models.keys()):
        formatted_benchmark = format_results_mlp(benchmark.results, benchmark.classification)
        # TEMP
        #=======================================================================
        # formatted_benchmark = format_ensemble_results(formatted_benchmark, benchmark.results_ensemble, benchmark.classification, 'mlp')
        #=======================================================================
        formatted_benchmark = format_best_result(formatted_benchmark, benchmark.best_result, benchmark.classification, 'mlp')
    else:
        print('\n\t\t\t[format_benchmark, 1 of 2] Should not happen!')

    model_names_dict = get_model_names_dict(benchmark)
    
    #ensemble_names_dict = get_ensemble_names_dict(benchmark)
    # TEMP
    algo = None
    
    for key, value in formatted_benchmark.items():
        if 'evolution' in key:
            i = 0
            for tbl in value:
                if i == 0:
                    ext = 'mean'
                else:
                    ext = 'se'
                tbl.columns = relabel_model_names(tbl.columns, model_names_dict)
                path = os.path.join(output_path, key + '_' + ext + '.csv')
                tbl.to_csv(path)
                i += 1
        elif 'ensemble' in key:
            if 'slm' in key: 
                algo = 'SLM'
            elif 'mlp' in key: 
                algo = 'MLP'
            else:
                print('\n\t\t\t[format_benchmark, 2 of 2] Should not happen!')
            
            formatted_benchmark[key].columns = relabel_ensemble_model_names(value.columns, ensemble_names_dict, algo)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)
        elif 'best_result' in key:
            # TEMP
            formatted_benchmark[key].columns = relabel_best_model_names(value.columns, algo)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)
        else: 
            formatted_benchmark[key].columns = relabel_model_names(value.columns, model_names_dict)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)


def merge_best_results(path): 
    slm_best_result_testing_value = pd.read_csv(os.path.join(path, 'slm_best_result_testing_value.csv'))
    slm_best_result_testing_value = slm_best_result_testing_value.drop(slm_best_result_testing_value.columns[0], axis=1) 
    mlp_best_result_testing_value = pd.read_csv(os.path.join(path, 'mlp_best_result_testing_value.csv'))
    mlp_best_result_testing_value = mlp_best_result_testing_value.drop(mlp_best_result_testing_value.columns[0], axis=1)
    frames = [slm_best_result_testing_value, mlp_best_result_testing_value]
    
    # TEMP
    #===========================================================================
    # slm_ensemble_testing_value = pd.read_csv(os.path.join(path, 'slm_ensemble_testing_value.csv'))
    # slm_ensemble_testing_value = slm_ensemble_testing_value.drop(slm_ensemble_testing_value.columns[0], axis=1)
    #===========================================================================
    
    # TEMP
    #===========================================================================
    # mlp_ensemble_testing_value = pd.read_csv(os.path.join(path, 'mlp_ensemble_testing_value.csv'))
    # mlp_ensemble_testing_value = mlp_ensemble_testing_value.drop(mlp_ensemble_testing_value.columns[0], axis=1)
    #===========================================================================
    
    # TEMP
    #===========================================================================
    # frames = [slm_best_result_testing_value, mlp_best_result_testing_value, slm_ensemble_testing_value]
    #===========================================================================
    
    # TEMP
    #===========================================================================
    # frames = [slm_best_result_testing_value, mlp_best_result_testing_value, slm_ensemble_testing_value, mlp_ensemble_testing_value]
    #===========================================================================
    
    # merge the datasets 
    merged = pd.concat(frames, axis=1)
    merged.to_csv(os.path.join(path, 'best_results_testing_value.csv'))
    
    if path.endswith('c_cancer') or path.endswith('c_credit') or path.endswith('c_diabetes') or path.endswith('c_sonar') or path.endswith('c_android'):
        slm_best_result_testing_auroc = pd.read_csv(os.path.join(path, 'slm_best_result_testing_accuracy.csv'))
        slm_best_result_testing_auroc = slm_best_result_testing_auroc.drop(slm_best_result_testing_auroc.columns[0], axis=1) 
        mlp_best_result_testing_auroc = pd.read_csv(os.path.join(path, 'mlp_best_result_testing_accuracy.csv'))
        mlp_best_result_testing_auroc = mlp_best_result_testing_auroc.drop(mlp_best_result_testing_auroc.columns[0], axis=1)
        frames = [slm_best_result_testing_auroc, mlp_best_result_testing_auroc]
        
        # TEMP
        #=======================================================================
        # slm_ensemble_testing_auroc = pd.read_csv(os.path.join(path, 'slm_ensemble_testing_accuracy.csv'))
        # slm_ensemble_testing_auroc = slm_ensemble_testing_auroc.drop(slm_ensemble_testing_auroc.columns[0], axis=1)
        #=======================================================================
        
        # TEMP
        #=======================================================================
        # mlp_ensemble_testing_value = pd.read_csv(os.path.join(path, 'mlp_ensemble_testing_value.csv'))
        # mlp_ensemble_testing_value = mlp_ensemble_testing_value.drop(mlp_ensemble_testing_value.columns[0], axis=1)
        #=======================================================================
        
        # TEMP
        #=======================================================================
        # frames = [slm_best_result_testing_auroc, mlp_best_result_testing_auroc, slm_ensemble_testing_auroc]
        #=======================================================================
        
        # TEMP
        #=======================================================================
        # frames = [slm_best_result_testing_value, mlp_best_result_testing_value, slm_ensemble_testing_value, mlp_ensemble_testing_value]
        #=======================================================================
        
        # merge the datasets 
        merged = pd.concat(frames, axis=1)
        merged.to_csv(os.path.join(path, 'best_results_testing_auroc.csv'))


def _is_classification(benchmark):
    return benchmark.dataset_name[0] == 'c'


def get_ensemble_names_dict(benchmark):
    return {key: {'name_short': value['name_short'],
                    'name_long': value['name_long']} for key, value in benchmark.ensembles.items()}


def get_model_names_dict(benchmark):
    return {key: {'name_short': value['name_short'],
                  'name_long': value['name_long']} for key, value in benchmark.models.items()}
