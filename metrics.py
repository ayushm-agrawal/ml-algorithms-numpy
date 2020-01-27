import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score

class metrics:

    def __init__(self):
        pass

    def confusion_matrix(self, y_true, y_pred, labels=None):
        if not labels:
            labels = np.unique(y_true) # number of classes in the labels

        conf_matrix =  np.zeros((len(labels), len(labels))) # init the matrix with zeros
        print("Labels: {}".format(labels))
        for idx, pred_label in enumerate(y_pred):
            row = 0
            col = 0
            for i in range(len(labels)):
                if labels[i] == pred_label:
                    col = i 
                if labels[i] == y_true[idx]:
                    row = i                    
                
            conf_matrix[row][col] += 1

        self.conf_matrix = conf_matrix
        self.classes = labels
        return conf_matrix
    

    def precision_score(self):
        precision_arr = []
        for idx, row in enumerate(self.conf_matrix):
            print("Row: {}".format(row))
            true_count = self.conf_matrix[idx][idx]
            print(true_count)
            row_sum = np.sum(row)
            print(row_sum)
            precision_arr.append(true_count/row_sum)
        
        self.prec_score = precision_arr
        return self.prec_score
        
    def recall_score(self):
        recall_arr = []
        for idx, col in enumerate(self.conf_matrix.T):
            true_count = self.conf_matrix[idx][idx]
            col_sum = np.sum(col)
            recall_arr.append(true_count/col_sum)

        self.rec_score = recall_arr
        return self.rec_score

    def f1_score(self):
        self.f1 = 2 * (self.prec_score * self.rec_score) / (self.prec_score + self.rec_score)
        return self.f1

    def roc_curve(self):
        pass

    def roc_auc_curve(self):
        pass


# TEST

met = metrics()
y_true = [1, -1,  0,  0,  1, -1,  1,  0, -1,  0,  1, -1,  1,  0,  0, -1,  0]
y_prediction = [-1, -1,  1,  0,  0,  0,  0, -1,  1, -1,  1,  1,  0,  0,  1,  1, -1]
conf_matrix = met.confusion_matrix(y_true, y_prediction)
prec_score = met.precision_score()
print("Confusion Matrix")
print(conf_matrix)
print("SK Learn Confusion Matrix")
print(confusion_matrix(y_true, y_prediction, labels=[-1,0,1]))
print("==========================")
print("Precision Score")
print(prec_score)
print("SK Learn Precision Score")
print(precision_score(y_true, y_prediction, average=None))
