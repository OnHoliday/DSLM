from numpy import sqrt, mean, square, empty, where, sum, log
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


class Metric():

    greater_is_better = None

    @staticmethod
    def evaluate(prediction, target):
        pass


class RootMeanSquaredError(Metric):

    greater_is_better = False
    
    @staticmethod
    def evaluate(prediction, target):
        #=======================================================================
        # value = 0
        # for i in range(prediction.shape[0]):
        #     dif = prediction[i] - target[i]
        #     value += dif * dif
        # value /= prediction.shape[0]
        # value = sqrt(value)[0]
        # return value
        #=======================================================================
        return sqrt(mean(square(prediction - target)))


class WeightedRootMeanSquaredError(Metric):

    greater_is_better = False 
    
    def __init__(self, weight_vector):
        self.weight_vector = weight_vector

    def evaluate(self, prediction, target):
        """Calculates RMSE taking into account a weight vector"""
        return sqrt(mean(square((prediction - target) * self.weight_vector)))

class Cross_entropy(Metric):
    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target):
        N = prediction.shape[0]
        ce = -sum(target * log(prediction)) / N
        return ce

class Multiclass_Accuracy(Metric):
    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target):
        return accuracy_score(target, prediction)


class Accuracy(Metric):

    greater_is_better = True
    
    @staticmethod
    def evaluate(prediction, target):
        
        #=======================================================================
        # return accuracy_score(target, where(prediction >= 0.5, True, False))
        #=======================================================================
        return accuracy_score(where(target >= 0.5, True, False), where(prediction >= 0.5, True, False))


class AUROC(Metric):

    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target, bound=True):
        if bound:
            auroc_y_score = empty(prediction.shape)
            for i in range(prediction.shape[0]):
                if prediction[i] < 0:
                    auroc_y_score[i] = 0
                elif prediction[i] > 1:
                    auroc_y_score[i] = 1
                else:
                    auroc_y_score[i] = prediction[i]
            return roc_auc_score(target, auroc_y_score)
        else:
            return roc_auc_score(target, prediction)


class AUROC_2(Metric):

    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target):
        return AUROC.evaluate(prediction, target, bound=False)


class BinaryCrossEntropy(Metric):
    
    greater_is_better = False
    
    @staticmethod
    def evaluate(prediction, target):
        y_pred_prob = empty(prediction.shape)
        for i in range(prediction.shape[0]):
            if prediction[i] < 0:
                y_pred_prob[i] = 0
            elif prediction[i] > 1:
                y_pred_prob[i] = 1
            else:
                y_pred_prob[i] = prediction[i]
        
        return log_loss(target, y_pred_prob)


def is_better(value_1, value_2, metric):
    #===========================================================================
    # print('[DEBUG] value 1 = %.5f, value 2 = %.5f\n' % (value_1, value_2))
    #===========================================================================
    if metric.greater_is_better:
        return value_1 > value_2
    else:
        return value_1 < value_2
