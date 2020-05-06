from algorithms.RegressionMultiTarget import multi_target_regression
from algorithms.RegressionSingleTarget import single_target_regression
from algorithms.MultiClassClassification import multiclass_classification
from algorithms.BinaryClassification import binary_classification

def main():
    multi_target_regression()
    multiclass_classification()
    
    single_target_regression()
    binary_classification()

main()