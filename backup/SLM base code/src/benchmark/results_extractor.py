from os import mkdir
from os.path import join, dirname, exists

from data.io_plm import read_csv_, get_results_folder
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

ALGOS_MAPPING = { 
    'slm_single': {
        'name-file': 'slm',  # best result
        'name-inside-file': 'SLM' 
    },
    'slm_fls_group': {
        'name-file': 'slm',
        'name-inside-file': 'SLM (FLS), SLM (FLS) + RST, SLM (FLS) + RWT'
    },
    'slm_ols_group': {
        'name-file': 'slm',
        'name-inside-file': 'SLM (OLS), SLM (OLS) + RST, SLM (OLS) + RWT'
    },
    'slm_fls_tie_edv_group': {
        'name-file': 'slm',
        'name-inside-file': 'SLM (FLS) + TIE, SLM (FLS) + EDV'
    },
    'slm_ols_edv': {
        'name-file': 'slm',
        'name-inside-file': 'SLM (OLS) + EDV'
    },
    'mlp_single': {
        'name-file': 'mlp',  # best result
        'name-inside-file': 'MLP'
    },
    'mlp_lbfgs': {
        'name-file': 'mlp',
        'name-inside-file': 'MLP (LBFGS)'
    },
    'mlp_adam': {
        'name-file': 'mlp',
        'name-inside-file': 'MLP (ADAM)'
    },
    'mlp_sgd': {
        'name-file': 'mlp',
        'name-inside-file': 'MLP (SGD)'
    },
    'slm_simple_ensemble': {
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM Simple Ensemble'
    },
    'slm_bagging_ensemble': {
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM Bagging Ensemble' 
    },
    'slm_riw_ensemble':{
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM RIW Ensemble'
    },
    'slm_boosting_1':{
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM Boosting Ensemble (Median + FLR)' 
    },
    'slm_boosting_2':{
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM Boosting Ensemble (Median + RLR)'
    },
    'slm_boosting_3':{
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM Boosting Ensemble (Mean + FLR)'
    },
    'slm_boosting_4':{
        'name-file': 'slm_ensemble',
        'name-inside-file': 'SLM Boosting Ensemble (Mean + RLR)' 
    },
    'mlp_simple_ensemble': {
        'name-file': 'mlp_ensemble',
        'name-inside-file': 'MLP Simple Ensemble'
    },
    'mlp_bagging_ensemble': {
        'name-file': 'mlp_ensemble',
        'name-inside-file': 'MLP Bagging Ensemble'
    },
    'mlp_boosting_1':{
        'name-file': 'mlp_ensemble',
        'name-inside-file': 'MLP Boosting Ensemble (Median + FLR)'
    } ,
    'mlp_boosting_2':{
        'name-file': 'mlp_ensemble',
        'name-inside-file': 'MLP Boosting Ensemble (Median + RLR)' 
    } ,
    'mlp_boosting_3':{
        'name-file': 'mlp_ensemble',
        'name-inside-file':  'MLP Boosting Ensemble (Mean + FLR)'
    },
    'mlp_boosting_4': {
        'name-file': 'mlp_ensemble',
        'name-inside-file': 'MLP Boosting Ensemble (Mean + RLR)'
    } 
}


def extract_results(path): 
    """ given a path, the method extracts the results inside that path""" 
    generate_test_boxplot(path, ['slm_single', 'mlp_single'], 'best_results_testing_value')
    generate_test_boxplot(path, ['mlp_boosting_4', 'slm_boosting_4'], 'testing_value')
    # generate_boxplot_error(path, 'testing_value')
    # generate_boxplot_error(path, 'training_value')
    # generate_boxplot_error(path, 'avg_inner_training_error')
    # generate_boxplot_error(path, 'avg_inner_validation_error')
    # generate_comparing_boxplot(path, 'avg_inner_validation_error', 'testing_value')


def generate_test_boxplot(path, algos_list, metric_name, labels_list=None, output_filename=None, min_max_y=None, ylabel='RMSE'):
    """generates test boxplot for a given list of algorithms""" 
    df_list = [] 
    if (metric_name != 'best_results_testing_value'):
        for algo in algos_list: 
            dataset_name = path.split('\\')[-1]
            
            to_read = join(path, metric_name + '.csv')
            
            # TEMP
            #===================================================================
            # name_file = ALGOS_MAPPING[algo]['name-file']
            # to_read = join(path, name_file + '_' + metric_name + '.csv')
            #===================================================================
            
            df = read_csv_(to_read)
            df = df[[ALGOS_MAPPING[algo]['name-inside-file']]]
            df_list.append(df)
        value = pd.concat(df_list, axis=1)
    elif (metric_name == 'best_results_testing_value'):
        for algo in algos_list: 
            dataset_name = path.split('\\')[-1]
            to_read = join(path, metric_name + '.csv')
            df = read_csv_(to_read)
            
            name = ALGOS_MAPPING[algo]['name-inside-file']
            if name not in df.keys():
                if name == 'SLM':
                    name = 'slm'
                elif name == 'MLP':
                    name = 'mlp'
                else:
                    print('\tUnknown name:', name)
            
            df = df[name]
            df_list.append(df)
        value = pd.concat(df_list, axis=1)
        
    if labels_list == None:
        labels_list = [ALGOS_MAPPING[algo]['name-inside-file'] for algo in algos_list]
    
    fig, ax = plt.subplots()
    boxplot = sns.boxplot(data=value, palette="PuBuGn_d")
    
    # print(boxplot.get_yaxis().get_view_interval())
    if min_max_y != None:
        boxplot.get_yaxis().set_view_interval(min_max_y[0], min_max_y[1])
    
    # boxplot.set(xlabel='Algorithms', ylabel='RMSE')
    boxplot.set(ylabel=ylabel)
    
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    sns.despine(left=True)
    
    # boxplot.set_xticklabels(labels_list, rotation=90) #changed from boxplot.get_xticklabels()-BE CAREFUL, SPECIFIC VALUES
    boxplot.set_xticklabels(labels_list)
    
    fig.set_size_inches(11.69, 8.27)
    results_folder = get_results_folder()
    if not exists(results_folder):
        mkdir(results_folder)
    results_folder_path = join(results_folder, dataset_name)
    if not exists(results_folder_path):
        mkdir(results_folder_path)
    
    if output_filename == None:
        output_filename = metric_name
    
    fig.savefig(join(results_folder_path, output_filename + '.svg'), bbox_inches='tight')
    fig.savefig(join(results_folder_path, output_filename + '.pdf'), bbox_inches='tight')


def generate_boxplot_error(path, metric_name):
    """generates a boxplot for a certain metric""" 
    dataset_name = path.split('\\')[-1]
    to_read = join(path, metric_name + '.csv')
    value = read_csv_(to_read)
    fig, ax = plt.subplots()
    boxplot = sns.boxplot(data=value, palette="PuBuGn_d")
    boxplot.set(xlabel='Algorithms', ylabel='RMSE')
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    sns.despine(left=True)
    boxplot.set_xticklabels(['SLM (FLS) Group', 'SLM (OLS) Group', 'SLM (FLS) + TIE/EDV', 'SLM (OLS) + EDV'], rotation=90)  # changed from boxplot.get_xticklabels()-BE CAREFUL, SPECIFIC VALUES
    fig.set_size_inches(11.69, 8.27)
    results_folder_path = join(get_results_folder(), dataset_name)
    fig.savefig(join(results_folder_path, metric_name + '.svg'), bbox_inches='tight')
    fig.savefig(join(results_folder_path, metric_name + '.pdf'), bbox_inches='tight')


def generate_comparing_boxplot(path, metric_one, metric_two):
    """generates a grouped boxplot taking into account two metrics"""
    dataset_name = path.split('\\')[-1]
    metric_one_path = join(path, metric_one + '.csv')
    metric_one_dataset = read_csv_(metric_one_path)
    metric_two_path = join(path, metric_two + '.csv')
    metric_two_dataset = read_csv_(metric_two_path)

    # melt datasets
    metric_one_dataset_long = metric_one_dataset.melt(var_name='algorithm', value_name=metric_one)
    metric_two_dataset_long = metric_two_dataset.melt(var_name='algorithm', value_name=metric_two)
    metric_two_dataset_long = metric_two_dataset_long.drop(metric_two_dataset_long.columns[0], axis=1)
    # concatenate datasets 
    concatenated_metrics = pd.concat([metric_one_dataset_long, metric_two_dataset_long], sort=False, axis=1)

    # melt concatenated dataset
    melted = pd.melt(concatenated_metrics, id_vars=['algorithm'], value_vars=[metric_one, metric_two], var_name='Metric')

    catplot = sns.catplot('algorithm', hue='Metric', y='value', data=melted, kind="box", legend=False, palette="PuBuGn_d",
                     height=5, aspect=1.75)
    catplot.set(xlabel='Algorithms', ylabel="RMSE")
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    sns.despine(left=True)
    catplot.set_xticklabels(['SLM (FLS) Group', 'SLM (OLS) Group', 'SLM (FLS) + TIE/EDV', 'SLM (OLS) + EDV'], rotation=90)  # BE CAREFUL, SPECIFIC VALUES
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Metric", frameon=False)  # this changes the legend to outside the plot
    # plt.figure()
    catplot.fig.set_size_inches(11.69, 8.27)
    results_folder_path = join(get_results_folder(), dataset_name)
    catplot.fig.savefig(join(results_folder_path, metric_one + '__' + metric_two + '.svg'), bbox_inches='tight')
    catplot.fig.savefig(join(results_folder_path, metric_one + '__' + metric_two + '.pdf'), bbox_inches='tight')

"""
#change working directory
os.chdir("C:\\Users\\Marta\\Documents\\GitHub\\pythonic-learning-machine\\datasets\\06_formatted\\c_ionosphere\\")

#testing value 
testing_value = pd.read_csv("testing_value.csv")
testing_value = testing_value[['SLM (Ensemble)', 'SLM (Ensemble) + RST', 'SLM (Ensemble) + RWT', 'SLM (Ensemble-Bagging)', 
                               'SLM (Ensemble-Bagging) + RST', 'SLM (Ensemble-Bagging) + RWT', 'SLM (Ensemble-RIW)',
                                'SLM (Ensemble-Boosting)', 'SLM (OLS)', 'SLM (OLS) + RST', 'SLM (OLS) + RWT']]
testing_value.head()

fig, ax = plt.subplots()
boxplot = sns.boxplot(data=testing_value, palette="PuBuGn_d")
boxplot.set(xlabel='Algorithms', ylabel='RMSE')
sns.set_context("paper", font_scale=1.5)
sns.set_style("white")
sns.despine(left=True)
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=90)
fig.set_size_inches(11.69, 8.27)
fig.savefig("testing_error.svg", bbox_inches='tight')
fig.savefig("testing_error.pdf", bbox_inches='tight')
plt.show()
"""
