from benchmark.new_benchmarker import Benchmarker
from benchmark.new_configuration import SLM_OLS, DSLM_OLS
from benchmark.training_benchmarker import TrainingBenchmarker
from metric import Cross_entropy, Multiclass_Accuracy

def generalization_benchmark2(dataset_name):
    models_to_run = DSLM_OLS
    benchmarker = Benchmarker(dataset_name, models=models_to_run, benchmark_id='slm_generalization', file_path='D:\SLM_CNN\cifar-10-batches-py', learning_metric=Multiclass_Accuracy, selection_metric=Multiclass_Accuracy)
    benchmarker.run_nested_cv()


# def generalization_benchmark(dataset_name):
#     models_to_run = SLM_OLS
#     benchmarker = Benchmarker(dataset_name, models=models_to_run, benchmark_id='slm_generalization', file_path='D:\SLM_CNN\cifar-10-batches-py')
#     benchmarker.run_nested_cv()

# def training_benchmark2(dataset_name):
#     models_to_run = SLM_OLS
#     benchmarker = TrainingBenchmarker(dataset_name, models=models_to_run, benchmark_id='slm_training')
#     benchmarker.run_nested_cv()
#
# def training_benchmark(dataset_name):
#     models_to_run = DSLM_OLS
#     benchmarker = TrainingBenchmarker(dataset_name, models=models_to_run, benchmark_id='slm_training')
#     benchmarker.run_nested_cv()


if __name__ == '__main__':
    
    # dataset = 'c_credit'
    dataset = 'data_batch_1'
    """ Generalization benchmark on a classification dataset """
    generalization_benchmark2(dataset)
    """ Training benchmark on a classification dataset """
    # training_benchmark(dataset)
    #
    # dataset = 'r_ppb'
    # """ Generalization benchmark on a regression dataset """
    # generalization_benchmark(dataset)
    # """ Training benchmark on a regression dataset """
    # training_benchmark(dataset)
