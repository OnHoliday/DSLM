""" Classification: 58 files (26 SLM, 31 MLP, 1 other) """
""" Regression: 46 files (20 SLM, 25 MLP, 1 other) """
"""
	best_results_testing_value.csv
	
	
		done =====> mlp_activation_function_frequency.csv
		done =====> mlp_alpha.csv
		out =====> mlp_avg_inner_training_error.csv
		done =====> mlp_avg_inner_validation_error.csv
		done =====> mlp_batch_size.csv
		(for now) out =====> mlp_best_configuration.csv (best configuration for each algorithm variant)
		done =====> mlp_best_overall_configuration_frequency.csv
		(for now) out =====> mlp_best_result_configuration.csv (best overall configuration saved as a dictionary)
		(for now) out =====> mlp_best_result_testing_accuracy.csv
		(for now) out =====> mlp_best_result_testing_value.csv
		out =====> mlp_best_result_training_accuracy.csv
		out =====> mlp_best_result_training_value.csv
		done =====> mlp_beta_1.csv
		done =====> mlp_beta_2.csv
		out =====> mlp_ensemble_base_algorithm.csv
		(for now) out =====> mlp_ensemble_testing_accuracy.csv
		out =====> mlp_ensemble_testing_value.csv (see best_results_testing_value.csv instead)
		out =====> mlp_ensemble_training_accuracy.csv
		out =====> mlp_ensemble_training_value.csv
		done =====> mlp_learning_rate.csv
		done =====> mlp_momentum.csv
		done =====> mlp_nesterovs_momentum.csv
		done =====> mlp_number_iterations.csv
		done =====> mlp_number_layers.csv
		done =====> mlp_number_neurons.csv
		done =====> mlp_penalty_frequency.csv
		done =====> mlp_shuffle.csv
		(for now) out =====> mlp_testing_accuracy.csv
		(for now) out =====> mlp_testing_value.csv
		out =====> mlp_training_accuracy.csv
		out =====> mlp_training_value.csv
	
	
		out =====> slm_avg_inner_training_error.csv
		done =====> slm_avg_inner_validation_error.csv
		(for now) out =====> slm_best_configuration.csv (best configuration for each algorithm variant)
		done =====> slm_best_overall_configuration_frequency.csv
		(for now) out =====> slm_best_result_configuration.csv (best overall configuration saved as a dictionary)
		(for now) out =====> slm_best_result_testing_accuracy.csv
		(for now) out =====> slm_best_result_testing_value.csv
		out =====> slm_best_result_training_accuracy.csv
		out =====> slm_best_result_training_value.csv
		out =====> slm_ensemble_base_algorithm.csv
		(for now) out =====> slm_ensemble_testing_accuracy.csv
		out =====> slm_ensemble_testing_value.csv (see best_results_testing_value.csv instead)
		out =====> slm_ensemble_training_accuracy.csv
		out =====> slm_ensemble_training_value.csv
		done =====> slm_learning_step_value.csv
		done =====> slm_number_generations.csv
		done =====> slm_number_layers.csv
		(for now) out =====> slm_processing_time.csv
		done =====> slm_RST_RWT_frequency.csv
		done =====> slm_subset_ratio.csv
		(for now) out =====> slm_testing_accuracy.csv
		(for now) out =====> slm_testing_value.csv
		done =====> slm_TIE_EDV_frequency.csv
		out =====> slm_training_accuracy.csv
		(for now) out =====> slm_training_time.csv
		out =====> slm_training_value.csv
"""

from os import pardir, listdir, mkdir
from os.path import join, dirname, exists

from pandas import read_csv

from data.io_plm import get_results_folder


def get_output_file(output_filename):
	
	results_folder = get_results_folder()
	
	if not exists(results_folder):
		mkdir(results_folder)
	latex_tables_path = join(results_folder, 'latex_tables')
	if not exists(latex_tables_path):
		mkdir(latex_tables_path)
	
	file_path = join(latex_tables_path, output_filename)
	return open(file_path, 'w')


def begin_table(output_file, caption='Some caption here', label='tab:some-label-here', two_columns=False):
	
	if two_columns:
		output_file.write('\\begin{table*}\n')
	else:
		output_file.write('\\begin{table}\n')
	output_file.write('\\centering\n')
	output_file.write('\\caption{%s}\n' % caption)
	output_file.write('\\label{%s}\n' % label)
	# output_file.write('%%\\resizebox{!}{18mm}{\n')
	# output_file.write('%\\scalebox{0.96}{\n')

"""
def get_labels(all_data):
	
	#rows = all_data[0].shape[0]
	methods = all_data[0].shape[1] - 1
	if methods == 4:
		labels = ['FLS variants', 'OLS variants', 'FLS + TIE/EDV', 'OLS + EDV']
		#labels = ['SLM-FLS variants', 'SLM-OLS variants', 'SLM-FLS + TIE/EDV', 'SLM-OLS + EDV']
		#labels = ['SLM group 1', 'SLM group 2', 'SLM group 3', 'SLM group 4']
	else:
		labels = ['BFGS', 'Adam', 'SGD']
		#labels = ['BFGS', 'SGD', 'Adam']
		#labels = ['MLP BFGS', 'MLP SGD', 'MLP Adam']
		#labels = ['MLP LBFGS', 'MLP SGD', 'MLP Adam']
	return labels
"""


def get_labels_from_names(names):
	
	labels = []

	for i in names:
		if i == 'SLM (FLS), SLM (FLS) + RST, SLM (FLS) + RWT':
			labels.append('FLS variants')
		elif i == 'SLM (OLS), SLM (OLS) + RST, SLM (OLS) + RWT':
			labels.append('OLS variants')
		elif i == 'SLM (FLS) + TIE, SLM (FLS) + EDV':
			labels.append('FLS + TIE/EDV')
		elif i == 'SLM (OLS) + EDV':
			labels.append('OLS + EDV')
		elif i == 'MLP (LBFGS)':
			labels.append('BFGS')
		elif i == 'MLP (ADAM)':
			labels.append('Adam')
		elif i == 'MLP (SGD)':
			labels.append('SGD')
		elif i == 'SLM':
			labels.append('SLM')
		elif i == 'MLP':
			labels.append('MLP')	
		elif i == 'SLM Simple Ensemble':
			labels.append('Simple')
		elif i == 'SLM Bagging Ensemble':
			labels.append('Bagging')
		elif i == 'SLM RIW Ensemble':
			labels.append('RIW')
		elif i == 'SLM Boosting Ensemble (Median + FLR)':
			labels.append('Boosting-1')
			# labels.append('SLM Boosting Median + FLR')
		elif i == 'SLM Boosting Ensemble (Median + RLR)':
			labels.append('Boosting-2')
			# labels.append('SLM Boosting Median + RLR')
		elif i == 'SLM Boosting Ensemble (Mean + FLR)':
			labels.append('Boosting-3')
			# labels.append('SLM Boosting Mean + FLR')
		elif i == 'SLM Boosting Ensemble (Mean + RLR)':
			labels.append('Boosting-4')
			# labels.append('SLM Boosting Mean + RLR')
		elif i == 'MLP Simple Ensemble':
			labels.append('Simple')
		elif i == 'MLP Bagging Ensemble':
			labels.append('Bagging')
		elif i == 'MLP Boosting Ensemble (Median + FLR)':
			labels.append('Boosting-1')
			# labels.append('MLP Boosting Median + FLR')
		elif i == 'MLP Boosting Ensemble (Median + RLR)':
			labels.append('Boosting-2')
			# labels.append('MLP Boosting Median + RLR')
		elif i == 'MLP Boosting Ensemble (Mean + FLR)':
			labels.append('Boosting-3')
			# labels.append('MLP Boosting Mean + FLR')
		elif i == 'MLP Boosting Ensemble (Mean + RLR)':
			labels.append('Boosting-4')
			# labels.append('MLP Boosting Mean + RLR')
		elif i == 'slm':
			labels.append('SLM')
		elif i == 'mlp':
			labels.append('MLP')	
		else:
			print('\t\tUnknown method with name:', i)

	return labels


def get_dataset_label_from_name(name):
	
	label = None
	
	if name == 'r_ppb':
		label = 'PPB'
	else:
		label = name[2:].capitalize()
		# print('\t\tUnknown dataset with name:', name)

	return label


def write_labels(output_file, labels):
	
	output_file.write('\\begin{tabular}{l')
	for label in labels:
		output_file.write('c')
	output_file.write('}\n')
	
	output_file.write('\\bf{Dataset}')
	for label in labels:
		output_file.write(' & \\bf{%s}' % label)
	output_file.write(' \\\\ \\hline\n')


def end_table(output_file, two_columns=False):
	
	output_file.write('\\end{tabular}\n')
	if two_columns:
		output_file.write('\\end{table*}\n')
	else:
		output_file.write('\\end{table}\n')


def process_validation_error(datasets_names, all_data, labels, output_file, caption='Validation error', label='tab:validation-error', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			if mean <= 100 and std <= 100:
				output_file.write(' & %.3f +- %.3f' % (mean, std))
			else:
				output_file.write(' & %.1e +- %.1e' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_best_overall_configuration_frequency(datasets_names, all_data, labels, output_file, caption='Best configuration by algorithm variant', label='tab:best-configuration-by-algorithm-variant', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			output_file.write(' & %d' % s)
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_number_of_iterations(datasets_names, all_data, labels, output_file, caption='Number of iterations', label='tab:number-of-iterations', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_number_of_iterations_max_count(maximum, datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = list(data[method_index + 1].astype(int))
			maximum_count = len([1 for i in s if i >= maximum])
			output_file.write(' & %d' % maximum_count)
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_number_of_layers(datasets_names, all_data, labels, output_file, caption='Number of layers', label='tab:number-of-layers', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_learning_step(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)

	
def process_subset_ratio(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[1]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			if method_index + 1 < data.shape[1]:
				s = data[method_index + 1].astype(float)
				
				import numpy as np
				s = s.replace(0, np.NaN)
				
				mean = s.mean()
				std = s.std()
				output_file.write(' & %.3f +- %.3f' % (mean, std))
			else:
				output_file.write(' & NA')
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_slm_fls_ssc(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)

	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{lcc}\n')
	output_file.write('\\bf{Dataset} & \\bf{EDV} & \\bf{TIE} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = data[method_index + 1].astype(int)
			output_file.write(' & %d & %d' % (s.loc[1], s.loc[2]))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
			
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_rst_rwt_use(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	
	output_file.write('\\begin{tabular}{l')
	for label in labels:
		output_file.write('ccc')
	output_file.write('}\n')
	
	output_file.write('\\multirow{2}{*}{\\bf{Dataset}}')
	for label in labels:
		output_file.write(' & \\multicolumn{3}{c}{\\bf{%s}}' % label)
	output_file.write(' \\\\ \\cline{2-7}\n')

	output_file.write('& \\bf{None} & \\bf{RST} & \\bf{RWT} & \\bf{None} & \\bf{RST} & \\bf{RWT} \\\\ \\hline\n')
	# output_file.write('& \\bf{No RST and RWT} & \\bf{RST} & \\bf{RWT} & \\bf{No RST and RWT} & \\bf{RST} & \\bf{RWT} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = data[method_index + 1].astype(int)
			output_file.write(' & %d & %d & %d' % (s.loc[1], s.loc[2], s.loc[3]))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\cline{2-7}')
			
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_number_of_neurons(datasets_names, all_data, labels, output_file, caption='Total number of hidden neurons', label='tab:total-number-of-hidden-neurons', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_activation_functions(datasets_names, all_data, labels, output_file, caption='Activation functions', label='tab:activation-functions', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	
	output_file.write('\\begin{tabular}{l')
	for label in labels:
		output_file.write('ccc')
	output_file.write('}\n')
	
	output_file.write('\\multirow{2}{*}{\\bf{Dataset}}')
	for label in labels:
		output_file.write(' & \\multicolumn{3}{c}{\\bf{%s}}' % label)
	output_file.write(' \\\\ \\cline{2-10}\n')

	output_file.write('& \\bf{Logistic} & \\bf{Relu} & \\bf{Tanh} & \\bf{Logistic} & \\bf{Relu} & \\bf{Tanh} & \\bf{Logistic} & \\bf{Relu} & \\bf{Tanh} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = data[method_index + 1].astype(int)
			output_file.write(' & %d & %d & %d' % (s.loc[1], s.loc[2], s.loc[3]))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\cline{2-10}')
			
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)

	
def process_penalty_frequency(datasets_names, all_data, labels, output_file, caption='L2 penalty frequency', label='tab:L2-penalty-frequency', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{l')
	for label in labels:
		output_file.write('cc')
	output_file.write('}\n')
	
	output_file.write('\\multirow{2}{*}{\\bf{Dataset}}')
	for label in labels:
		output_file.write(' & \\multicolumn{2}{c}{\\bf{%s}}' % label)
	output_file.write(' \\\\ \\cline{2-7}\n')

	output_file.write('& \\bf{With L2 penalty} & \\bf{Without L2 penalty} & \\bf{With L2 penalty} & \\bf{Without L2 penalty} & \\bf{With L2 penalty} & \\bf{Without L2 penalty} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = data[method_index + 1].astype(int)
			output_file.write(' & %d & %d' % (s.loc[1], s.loc[2]))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\cline{2-7}')
			
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_alpha(datasets_names, all_data, labels, output_file, caption='L2 penalty', label='tab:L2-penalty', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			
			import numpy as np
			s = s.replace(0, np.NaN)
			
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_learning_rate(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_batch_size(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_batch_shuffle(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{l')
	for label in labels:
		output_file.write('cc')
	output_file.write('}\n')
	
	output_file.write('\\multirow{2}{*}{\\bf{Dataset}}')
	for label in labels:
		output_file.write(' & \\multicolumn{2}{c}{\\bf{%s}}' % label)
	output_file.write(' \\\\ \\cline{2-5}\n')

	output_file.write('& \\bf{With shuffle} & \\bf{Without shuffle} & \\bf{With shuffle} & \\bf{Without shuffle} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = list(data[method_index + 1].astype(str))
			true_count = len([1 for i in s if i == 'True'])
			false_count = len(s) - true_count
			output_file.write(' & %d & %d' % (true_count, false_count))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\cline{2-5}')
			
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


# SGD
def process_momentum(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{lc}\n')
	output_file.write('\\bf{Dataset} & \\bf{Momentum} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


# (bool) SGD
def process_nesterovs_momentum(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{lcc}\n')
	output_file.write('\\bf{Dataset} & \\bf{With} & \\bf{Without} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			
			s = list(data[method_index + 1].astype(str))
			true_count = len([1 for i in s if i == 'True'])
			false_count = len(s) - true_count
			output_file.write(' & %d & %d' % (true_count, false_count))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
			
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


# Adam
def process_beta_1(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{lc}\n')
	output_file.write('\\bf{Dataset} & \\bf{Beta 1} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


# Adam
def process_beta_2(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]

	output_file.write('\\begin{tabular}{lc}\n')
	output_file.write('\\bf{Dataset} & \\bf{Beta 2} \\\\ \\hline\n')
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_generalization_slm_vs_mlp_no_ensemble(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	start = 0
	end = 2
	
	""" meh ... """
	labels = labels[0][start:end]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(start, end):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			if mean <= 100 and std <= 100:
				output_file.write(' & %.3f +- %.3f' % (mean, std))
			else:
				output_file.write(' & %.1e +- %.1e' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_generalization_slm_ensembles(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	start = 2
	end = 9
	
	""" meh ... """
	labels = labels[0][start:end]
	write_labels(output_file, labels)

	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(start, end):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			if mean <= 100 and std <= 100:
				output_file.write(' & %.3f +- %.3f' % (mean, std))
			else:
				output_file.write(' & %.1e +- %.1e' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_generalization_mlp_ensembles(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	start = 9
	end = 15

	""" meh ... """
	labels = labels[0][start:end]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(start, end):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			if mean <= 100 and std <= 100:
				output_file.write(' & %.3f +- %.3f' % (mean, std))
			else:
				output_file.write(' & %.1e +- %.1e' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_generalization(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			if mean <= 100 and std <= 100:
				output_file.write(' & %.3f +- %.3f' % (mean, std))
			else:
				output_file.write(' & %.1e +- %.1e' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def process_auroc(datasets_names, all_data, labels, output_file, caption='Some caption ...', label='tab:some-label', two_columns=False):
	
	begin_table(output_file, caption, label, two_columns)
	
	""" meh ... """
	labels = labels[0]
	write_labels(output_file, labels)
	
	for dataset_index in range(len(datasets_names)):
		output_file.write('\\bf{%s}' % get_dataset_label_from_name(datasets_names[dataset_index]))
		
		data = all_data[dataset_index]
		for method_index in range(len(labels)):
			s = data[method_index + 1].astype(float)
			mean = s.mean()
			std = s.std()
			output_file.write(' & %.3f +- %.3f' % (mean, std))
		
		if dataset_index != len(datasets_names) - 1:
			output_file.write(' \\\\ \\hline')
		output_file.write('\n')
		
	end_table(output_file, two_columns=two_columns)


def get_data_and_labels(datasets_folders, files):
	all_data = {}
	all_labels = {}
	
	for dataset_folder in datasets_folders:
		
		for file_name in files:
			data = read_csv(join(dataset_folder, file_name), header=None)
			
			first_row = data.iloc(0)[0]
			names = list(first_row[1 : ])
			labels = get_labels_from_names(names)
			
			data = data[1 : data.shape[0]]
			
			if file_name in all_data:
				all_data[file_name].append(data)
				all_labels[file_name].append(labels)
			else:
				all_data[file_name] = [data]
				all_labels[file_name] = [labels]
				
	return all_data, all_labels


def slm_groups(datasets_names, datasets_folders):
	
	# e.g., data = read_csv('..\\datasets\\06_formatted\\c_diabetes\\slm_avg_inner_validation_error.csv', header=None)
	files = ['slm_avg_inner_validation_error.csv']
	files += ['slm_best_overall_configuration_frequency.csv']
	files += ['slm_number_generations.csv']
	files += ['slm_number_layers.csv']	
	# files += ['slm_TIE_EDV_frequency.csv']
	# files += ['slm_subset_ratio.csv']
	# files += ['slm_RST_RWT_frequency.csv']
	files += ['slm_learning_step_value.csv']
	
	all_data, all_labels = get_data_and_labels(datasets_folders, files)
	
	classification_datasets_files = ['slm_testing_accuracy.csv']
	# classification_datasets_files += ['slm_ensemble_testing_accuracy.csv']
	classification_datasets_files += ['slm_best_result_testing_accuracy.csv']
	
	# classification_datasets_folders = datasets_folders[:4]
	# classification_datasets_names = ['c_cancer', 'c_credit', 'c_diabetes', 'c_sonar']
	# TEMP
	classification_datasets_folders = datasets_folders
	# classification_datasets_names = ['c_diabetes']
	classification_datasets_names = ['c_android']
	
	classification_data, classification_labels = get_data_and_labels(classification_datasets_folders, classification_datasets_files)
	
	output_filename = 'slm_testing_auroc.txt'
	output_file = get_output_file(output_filename)
	caption = 'Testing AUROC for each SLM variant considered'
	label = 'tab:testing-auroc-per-slm-variant'
	data = classification_data['slm_testing_accuracy.csv']
	labels = classification_labels['slm_testing_accuracy.csv']
	process_auroc(classification_datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'slm_best_result_testing_auroc.txt'
	output_file = get_output_file(output_filename)
	caption = 'Testing AUROC for SLM'
	label = 'tab:testing-auroc-slm'
	data = classification_data['slm_best_result_testing_accuracy.csv']
	labels = classification_labels['slm_best_result_testing_accuracy.csv']
	process_auroc(classification_datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()

	output_filename = 'slm_validation_error.txt'
	output_file = get_output_file(output_filename)
	caption = 'Validation error for each SLM variant considered'
	label = 'tab:validation-error-per-slm-variant'
	data = all_data['slm_avg_inner_validation_error.csv']
	labels = all_labels['slm_avg_inner_validation_error.csv']
	process_validation_error(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'slm_best_configuration.txt'
	output_file = get_output_file(output_filename)
	caption = 'Best SLM configuration by variant'
	label = 'tab:' + caption.replace(' ', '-').lower()
	data = all_data['slm_best_overall_configuration_frequency.csv']
	labels = all_labels['slm_best_overall_configuration_frequency.csv']
	process_best_overall_configuration_frequency(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'slm_number_of_iterations.txt'
	output_file = get_output_file(output_filename)
	caption = 'Number of iterations for each SLM variant considered'
	label = 'tab:number-of-iterations-per-slm-variant'
	data = all_data['slm_number_generations.csv']
	labels = all_labels['slm_number_generations.csv']
	process_number_of_iterations(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'slm_number_of_iterations_max_count.txt'
	output_file = get_output_file(output_filename)
	caption = '[Internal results] Number of times that the maximum number of iterations was reached'
	label = 'tab:slm-number-of-iterations-max-count'
	data = all_data['slm_number_generations.csv']
	labels = all_labels['slm_number_generations.csv']
	process_number_of_iterations_max_count(100, datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'slm_number_of_layers.txt'
	output_file = get_output_file(output_filename)
	caption = 'Number of layers for each SLM variant considered'
	label = 'tab:number-of-layers-per-slm-variant'
	data = all_data['slm_number_layers.csv']
	labels = all_labels['slm_number_layers.csv']
	process_number_of_layers(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'slm_learning_step_value.txt'
	output_file = get_output_file(output_filename)
	caption = 'Learning step for each SLM-FLS variant considered'
	label = 'tab:learning-step-per-slm-fls-variant'
	data = all_data['slm_learning_step_value.csv']
	labels = all_labels['slm_learning_step_value.csv']
	process_learning_step(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	#===========================================================================
	# output_filename = 'slm_subset_ratio.txt'
	# output_file = get_output_file(output_filename)
	# caption = 'Subset ratio for each SLM variant considered'
	# label = 'tab:subset-ratio-per-slm-variant'
	# data = all_data['slm_subset_ratio.csv']
	# labels = all_labels['slm_subset_ratio.csv']
	# process_subset_ratio(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	# output_file.close()
	#===========================================================================
	
	#===========================================================================
	# output_filename = 'slm_TIE_EDV_frequency.txt'
	# output_file = get_output_file(output_filename)
	# caption = 'EDV and TIE use in SLM-FLS'
	# label = 'tab:slm-fls-ssc'
	# data = all_data['slm_TIE_EDV_frequency.csv']
	# labels = all_labels['slm_TIE_EDV_frequency.csv']
	# process_slm_fls_ssc(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	# output_file.close()
	#===========================================================================
	
	#===========================================================================
	# output_filename = 'slm_RST_RWT_frequency.txt'
	# output_file = get_output_file(output_filename)
	# caption = 'RST and RWT use in the FLS and the OLS variants'
	# label = 'tab:rst-rwt-use'
	# data = all_data['slm_RST_RWT_frequency.csv']
	# labels = all_labels['slm_RST_RWT_frequency.csv']
	# process_rst_rwt_use(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	# output_file.close()
	#===========================================================================


def mlp_groups(datasets_names, datasets_folders):
	
	files = ['mlp_avg_inner_validation_error.csv']
	files += ['mlp_best_overall_configuration_frequency.csv']
	files += ['mlp_number_iterations.csv']
	files += ['mlp_number_layers.csv']
	files += ['mlp_number_neurons.csv']
	files += ['mlp_activation_function_frequency.csv']
	files += ['mlp_penalty_frequency.csv']
	files += ['mlp_alpha.csv']
	files += ['mlp_learning_rate.csv']
	files += ['mlp_batch_size.csv']
	files += ['mlp_shuffle.csv']
	files += ['mlp_momentum.csv']
	files += ['mlp_nesterovs_momentum.csv']
	files += ['mlp_beta_1.csv']
	files += ['mlp_beta_2.csv']
	
	all_data, all_labels = get_data_and_labels(datasets_folders, files)
	
	classification_datasets_files = ['mlp_testing_accuracy.csv']
	# classification_datasets_files += ['mlp_ensemble_testing_accuracy.csv']
	classification_datasets_files += ['mlp_best_result_testing_accuracy.csv']
	
	# classification_datasets_folders = datasets_folders[:4]
	# classification_datasets_names = ['c_cancer', 'c_credit', 'c_diabetes', 'c_sonar']
	# TEMP
	classification_datasets_folders = datasets_folders
	# classification_datasets_names = ['c_diabetes']
	classification_datasets_names = ['c_android']
	
	classification_data, classification_labels = get_data_and_labels(classification_datasets_folders, classification_datasets_files)
	
	output_filename = 'mlp_testing_auroc.txt'
	output_file = get_output_file(output_filename)
	caption = 'Testing AUROC for each MLP variant considered'
	label = 'tab:testing-auroc-per-mlp-variant'
	data = classification_data['mlp_testing_accuracy.csv']
	labels = classification_labels['mlp_testing_accuracy.csv']
	process_auroc(classification_datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_best_result_testing_auroc.txt'
	output_file = get_output_file(output_filename)
	caption = 'Testing AUROC for MLP'
	label = 'tab:testing-auroc-mlp'
	data = classification_data['mlp_best_result_testing_accuracy.csv']
	labels = classification_labels['mlp_best_result_testing_accuracy.csv']
	process_auroc(classification_datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_validation_error.txt'
	output_file = get_output_file(output_filename)
	caption = 'Validation error for each MLP variant considered'
	label = 'tab:validation-error-per-mlp-variant'
	data = all_data['mlp_avg_inner_validation_error.csv']
	labels = all_labels['mlp_avg_inner_validation_error.csv']
	process_validation_error(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_best_configuration.txt'
	output_file = get_output_file(output_filename)
	caption = 'Best MLP configuration by variant'
	label = 'tab:' + caption.replace(' ', '-').lower()
	data = all_data['mlp_best_overall_configuration_frequency.csv']
	labels = all_labels['mlp_best_overall_configuration_frequency.csv']
	process_best_overall_configuration_frequency(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_number_of_iterations.txt'
	output_file = get_output_file(output_filename)
	caption = 'Number of iterations for each MLP variant considered'
	label = 'tab:number-of-iterations-per-mlp-variant'
	data = all_data['mlp_number_iterations.csv']
	labels = all_labels['mlp_number_iterations.csv']
	process_number_of_iterations(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_number_of_iterations_max_count.txt'
	output_file = get_output_file(output_filename)
	caption = '[Internal results] Number of times that the maximum number of iterations was reached'
	label = 'tab:mlp-number-of-iterations-max-count'
	data = all_data['mlp_number_iterations.csv']
	labels = all_labels['mlp_number_iterations.csv']
	process_number_of_iterations_max_count(100, datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()

	output_filename = 'mlp_number_of_layers.txt'
	output_file = get_output_file(output_filename)
	caption = 'Number of layers for each MLP variant considered'
	label = 'tab:number-of-layers-per-mlp-variant'
	data = all_data['mlp_number_layers.csv']
	labels = all_labels['mlp_number_layers.csv']
	process_number_of_layers(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_number_of_neurons.txt'
	output_file = get_output_file(output_filename)
	caption = 'Total number of hidden neurons for each MLP variant considered'
	label = 'tab:total-number-of-hidden-neurons-per-mlp-variant'
	data = all_data['mlp_number_neurons.csv']
	labels = all_labels['mlp_number_neurons.csv']
	process_number_of_neurons(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()

	output_filename = 'mlp_activation_function_frequency.txt'
	output_file = get_output_file(output_filename)
	caption = 'Activation functions use by MLP variant'
	label = 'tab:activation-functions-by-mlp-variant'
	data = all_data['mlp_activation_function_frequency.csv']
	labels = all_labels['mlp_activation_function_frequency.csv']
	process_activation_functions(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_penalty_frequency.txt'
	output_file = get_output_file(output_filename)
	caption = 'L2 penalty use by MLP variant'
	label = 'tab:L2-penalty-use'
	data = all_data['mlp_penalty_frequency.csv']
	labels = all_labels['mlp_penalty_frequency.csv']
	process_penalty_frequency(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()

	output_filename = 'mlp_alpha.txt'
	output_file = get_output_file(output_filename)
	caption = 'L2 penalty by MLP variant'
	label = 'tab:L2-penalty'
	data = all_data['mlp_alpha.csv']
	labels = all_labels['mlp_alpha.csv']
	process_alpha(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()

	output_filename = 'mlp_learning_rate.txt'
	output_file = get_output_file(output_filename)
	caption = 'Learning rate by MLP variant'
	label = 'tab:learning-rate'
	data = all_data['mlp_learning_rate.csv']
	labels = all_labels['mlp_learning_rate.csv']
	process_learning_rate(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_batch_size.txt'
	output_file = get_output_file(output_filename)
	caption = 'Batch size by MLP variant'
	label = 'tab:batch-size'
	data = all_data['mlp_batch_size.csv']
	labels = all_labels['mlp_batch_size.csv']
	process_batch_size(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_shuffle.txt'
	output_file = get_output_file(output_filename)
	caption = 'Batch shuffle use by MLP variant'
	label = 'tab:batch-shuffle'
	data = all_data['mlp_shuffle.csv']
	labels = all_labels['mlp_shuffle.csv']
	process_batch_shuffle(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	output_filename = 'mlp_momentum.txt'
	output_file = get_output_file(output_filename)
	caption = 'Momentum in MLP SGD'
	label = 'tab:momentum'
	data = all_data['mlp_momentum.csv']
	labels = all_labels['mlp_momentum.csv']
	process_momentum(datasets_names, data, labels, output_file, caption, label, two_columns=False)
	output_file.close()
	
	output_filename = 'mlp_nesterovs_momentum.txt'
	output_file = get_output_file(output_filename)
	caption = 'Nesterovs momentum use in MLP SGD'
	label = 'tab:nesterovs-momentum'
	data = all_data['mlp_nesterovs_momentum.csv']
	labels = all_labels['mlp_nesterovs_momentum.csv']
	process_nesterovs_momentum(datasets_names, data, labels, output_file, caption, label, two_columns=False)
	output_file.close()
	
	output_filename = 'mlp_beta_1.txt'
	output_file = get_output_file(output_filename)
	caption = 'Beta 1 in MLP Adam'
	label = 'tab:beta-1'
	data = all_data['mlp_beta_1.csv']
	labels = all_labels['mlp_beta_1.csv']
	process_beta_1(datasets_names, data, labels, output_file, caption, label, two_columns=False)
	output_file.close()
	
	output_filename = 'mlp_beta_2.txt'
	output_file = get_output_file(output_filename)
	caption = 'Beta 2 in MLP Adam'
	label = 'tab:beta-2'
	data = all_data['mlp_beta_2.csv']
	labels = all_labels['mlp_beta_2.csv']
	process_beta_2(datasets_names, data, labels, output_file, caption, label, two_columns=False)
	output_file.close()


def process_inner(datasets_names, datasets_folders):
		
	slm_groups(datasets_names, datasets_folders)
	mlp_groups(datasets_names, datasets_folders)
	
	files = ['best_results_testing_value.csv']
	files += ['best_results_testing_auroc.csv']
	
	all_data, all_labels = get_data_and_labels(datasets_folders, files)
	
	# TEMP
	#===========================================================================
	# # output_filename = 'generalization_slm_vs_mlp_no_ensemble.txt'
	# output_filename = 'generalization_slm_vs_mlp.txt'
	# output_file = get_output_file(output_filename)
	# caption = 'Generalization of SLM and MLP'
	# label = 'tab:generalization-slm-vs-mlp'
	# data = all_data['best_results_testing_value.csv']
	# labels = all_labels['best_results_testing_value.csv']
	# process_generalization_slm_vs_mlp_no_ensemble(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	# output_file.close()
	#===========================================================================
	
	# output_filename = 'generalization_slm_vs_mlp_no_ensemble.txt'
	output_filename = 'generalization_slm_vs_mlp.txt'
	output_file = get_output_file(output_filename)
	caption = 'Generalization of SLM and MLP'
	label = 'tab:generalization-slm-vs-mlp'
	data = all_data['best_results_testing_auroc.csv']
	labels = all_labels['best_results_testing_auroc.csv']
	process_generalization_slm_vs_mlp_no_ensemble(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	output_file.close()
	
	#===========================================================================
	# output_filename = 'generalization_slm_ensembles.txt'
	# output_file = get_output_file(output_filename)
	# caption = 'Generalization of SLM ensembles'
	# label = 'tab:generalization-slm-ensembles'
	# data = all_data['best_results_testing_value.csv']
	# labels = all_labels['best_results_testing_value.csv']
	# process_generalization_slm_ensembles(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	# output_file.close()
	# 
	# output_filename = 'generalization_mlp_ensembles.txt'
	# output_file = get_output_file(output_filename)
	# caption = 'Generalization of MLP ensembles'
	# label = 'tab:generalization-mlp-ensembles'
	# data = all_data['best_results_testing_value.csv']
	# labels = all_labels['best_results_testing_value.csv']
	# process_generalization_mlp_ensembles(datasets_names, data, labels, output_file, caption, label, two_columns=True)
	# output_file.close()
	#===========================================================================
	
	#------------------------------------ output_filename = 'generalization.txt'
	#---------------------------------- output_file = get_output_file(output_filename)
	#-------------------------------------------- caption = 'Generalization ...'
	#-------------------------------------------- label = 'tab:generalization-4'
	#------------------------- data = all_data['best_results_testing_value.csv']
	#--------------------- labels = all_labels['best_results_testing_value.csv']
	# process_generalization(datasets_names, data, labels, output_file, caption, label, two_columns = True)
	#------------------------------------------------------- output_file.close()


def process_selected_datasets(datasets_names):
	
	datasets_folders = []
	
	base_folder = join(dirname(__file__), pardir, 'datasets', '06_formatted')
	
	for dataset_name in datasets_names:
		dataset_folder = join(base_folder, dataset_name)
		datasets_folders.append(dataset_folder)
	
	process_inner(datasets_names, datasets_folders)


def process_all():
	
	datasets_names = []
	datasets_folders = []
	
	base_folder = join(dirname(__file__), pardir, 'datasets', '06_formatted')
	
	for dataset_name in listdir(base_folder):
		datasets_names.append(dataset_name)
		dataset_folder = join(base_folder, dataset_name)
		datasets_folders.append(dataset_folder)
	
	process_inner(datasets_names, datasets_folders)


if __name__ == '__main__':
	
	process_all()
	
	# process_selected_datasets(datasets_names)
