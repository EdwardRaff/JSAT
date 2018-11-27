package jsat.classifiers;

import jsat.parallelization.ParallelizationEngine;

public interface ParallelizedClassifier extends Classifier {
	public void train(ClassificationDataSet dataSet, ParallelizationEngine engine);
	public CategoricalResults classify(DataPoint data, ParallelizationEngine engine);
}
