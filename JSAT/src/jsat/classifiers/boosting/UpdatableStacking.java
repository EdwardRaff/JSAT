
package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.classifiers.linear.LinearBatch;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.BaseUpdateableRegressor;
import jsat.regression.RegressionDataSet;
import jsat.regression.UpdateableRegressor;

/**
 * This provides an implementation of the Stacking ensemble method meant for 
 * Updatable models. Stacking learns several base classifiers and a top level 
 * classifier learns to predict the target based on the outputs of all the 
 * ensambled models. Historically a linear model (such as {@link LinearBatch}) 
 * is used, which translates to learning a weighted vote of the classifier 
 * outputs. However any classifier may be used so long as it supports the 
 * desired target type. <br>
 * <br>
 * Note, that Stacking tends to work best when the base classifiers produce 
 * reasonable probability estimates. <br>
 * Stacking supports {@link #supportsWeightedData() weighted data instances} if 
 * the aggregating model does. 
 * <br>
 * See: Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5, 241–259. 
 * 
 * @author Edward Raff
 */
public class UpdatableStacking implements UpdateableClassifier, UpdateableRegressor
{
    /*
     * TODO should investigate providing a 'skip' paramter, as the first few 
     * predictions from the base models are going to be rubbish. So let them 
     * settle in a littl ebefore we start updating the aggregator off their 
     * predictions
     */
    

	private static final long serialVersionUID = -5111303510263114862L;
	/**
     * The number of weights needed per model
     */
    private int weightsPerModel;
    private UpdateableClassifier aggregatingClassifier;
    private List<UpdateableClassifier> baseClassifiers;
    
    private UpdateableRegressor aggregatingRegressor;
    private List<UpdateableRegressor> baseRegressors;

    /**
     * Creates a new Stacking classifier
     * @param aggregatingClassifier the classifier used to merge the results of all the input classifiers
     * @param baseClassifiers the list of base classifiers to ensemble
     */
    public UpdatableStacking(UpdateableClassifier aggregatingClassifier, List<UpdateableClassifier> baseClassifiers)
    {
        if(baseClassifiers.size() < 2)
            throw new IllegalArgumentException("base classifiers must contain at least 2 elements, not " + baseClassifiers.size());
        
        this.aggregatingClassifier = aggregatingClassifier;
        this.baseClassifiers = baseClassifiers;
        
        boolean allRegressors = aggregatingClassifier instanceof UpdateableRegressor;
        for(UpdateableClassifier cl : baseClassifiers)
            if(!(cl instanceof UpdateableRegressor))
                allRegressors = false;
        
        if(allRegressors)
        {
            aggregatingRegressor = (UpdateableRegressor) aggregatingClassifier;
            baseRegressors = (List) baseClassifiers;//ugly type easure exploitation... 
        }
    }
    
    /**
     * Creates a new Stacking classifier.
     * @param aggregatingClassifier the classifier used to merge the results of all the input classifiers
     * @param baseClassifiers the array of base classifiers to ensemble
     */
    public UpdatableStacking(UpdateableClassifier aggregatingClassifier, UpdateableClassifier... baseClassifiers)
    {
        this(aggregatingClassifier, Arrays.asList(baseClassifiers));
    }
    
    /**
     * Creates a new Stacking regressor
     * @param aggregatingRegressor the regressor used to merge the results of all the input classifiers
     * @param baseRegressors the list of base regressors to ensemble
     */
    public UpdatableStacking(UpdateableRegressor aggregatingRegressor, List<UpdateableRegressor> baseRegressors)
    {
        this.aggregatingRegressor = aggregatingRegressor;
        this.baseRegressors = baseRegressors;
        
        boolean allClassifiers = aggregatingRegressor instanceof UpdateableClassifier;
        for(UpdateableRegressor reg : baseRegressors)
            if(!(reg instanceof UpdateableClassifier))
                allClassifiers = false;
        
        if(allClassifiers)
        {
            aggregatingClassifier = (UpdateableClassifier) aggregatingRegressor;
            baseClassifiers = (List) baseRegressors;//ugly type easure exploitation... 
        }
    }
    
    /**
     * Creates a new Stacking regressor.
     * @param aggregatingRegressor the regressor used to merge the results of all the input classifiers
     * @param baseRegressors the array of base regressors to ensemble
     */
    public UpdatableStacking(UpdateableRegressor aggregatingRegressor, UpdateableRegressor... baseRegressors)
    {
        this(aggregatingRegressor, Arrays.asList(baseRegressors));
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public UpdatableStacking(UpdatableStacking toCopy)
    {
        this.weightsPerModel = toCopy.weightsPerModel;
        if(toCopy.aggregatingClassifier != null)
        {
            this.aggregatingClassifier = toCopy.aggregatingClassifier.clone();
            this.baseClassifiers = new ArrayList<UpdateableClassifier>(toCopy.baseClassifiers.size());
            for(UpdateableClassifier bc : toCopy.baseClassifiers)
                this.baseClassifiers.add(bc.clone());
            
            if(toCopy.aggregatingRegressor == toCopy.aggregatingClassifier)//supports both
            {
                aggregatingRegressor = (UpdateableRegressor) aggregatingClassifier;
                baseRegressors = (List) baseClassifiers;//ugly type easure exploitation... 
            }
        }
        else//we are doing with regressors only
        {
            this.aggregatingRegressor = toCopy.aggregatingRegressor.clone();
            this.baseRegressors = new ArrayList<UpdateableRegressor>(toCopy.baseRegressors.size());
            for(UpdateableRegressor br : toCopy.baseRegressors)
                this.baseRegressors.add(br.clone());
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return aggregatingClassifier.classify(getPredVecC(data, 1.0));
    }

    /**
     * Gets the predicted vector wrapped in a new DataPoint from a data point 
     * assuming we are doing classification 
     * @param data the data point to get the classifier from
     * @param weight the weight to use for the data point object
     * @return the vector of predictions from each classifier
     */
    private DataPoint getPredVecC(DataPoint data, double weight)
    {
        Vec w = new DenseVector(weightsPerModel*baseClassifiers.size());
        if(weightsPerModel == 1)
            for(int i = 0; i < baseClassifiers.size(); i++)
                w.set(i, baseClassifiers.get(i).classify(data).getProb(0)*2-1);
        else
        {
            for(int i = 0; i < baseClassifiers.size(); i++)
            {
                CategoricalResults pred = baseClassifiers.get(i).classify(data);
                for(int j = 0; j < weightsPerModel; j++)
                    w.set(i*weightsPerModel+j, pred.getProb(j));
            }
                    
        }
        return new DataPoint(w, weight);
    }
    
    /**
     * Gets the predicted vector wrapped in a new DataPoint from a data point 
     * assuming we are doing regression 
     * @param data the data point to get the classifier from
     * @param weight the weight to use for the data point object
     * @return the vector of predictions from each regressor
     */
    private DataPoint getPredVecR(DataPoint data, double weight)
    {
        Vec w = new DenseVector(baseRegressors.size());
        for (int i = 0; i < baseRegressors.size(); i++)
            w.set(i, baseRegressors.get(i).regress(data));
        return new DataPoint(w, weight);
    }
    
    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        final int C = predicting.getNumOfCategories();
        weightsPerModel = C == 2 ? 1 : C;
        //set up all models, agregating gets different arugmetns since it gets the created input from the base models
        aggregatingClassifier.setUp(new CategoricalData[0], weightsPerModel*baseClassifiers.size(), predicting);
        for(UpdateableClassifier uc : baseClassifiers)
            uc.setUp(categoricalAttributes, numericAttributes, predicting);
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        //predate first, gives an unbiased udpdate for the aggregator
        aggregatingClassifier.update(getPredVecC(dataPoint, dataPoint.getWeight()), targetClass);
        //now update the base models
        for(UpdateableClassifier uc : baseClassifiers)
            uc.update(dataPoint, targetClass);
        
    }
    
    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes)
    {
        weightsPerModel = 1;
        aggregatingRegressor.setUp(new CategoricalData[0], weightsPerModel*baseRegressors.size());
        for(UpdateableRegressor ur : baseRegressors)
            ur.setUp(categoricalAttributes, numericAttributes);
    }

    @Override
    public void update(DataPoint dataPoint, double targetValue)
    {
        //predate first, gives an unbiased udpdate for the aggregator
        aggregatingRegressor.update(getPredVecR(dataPoint, dataPoint.getWeight()), targetValue);
        //now update the base models
        for(UpdateableRegressor ur : baseRegressors)
            ur.update(dataPoint, targetValue);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        BaseUpdateableClassifier.trainEpochs(dataSet, this, 1);
    }

    @Override
    public boolean supportsWeightedData()
    {
        if(aggregatingClassifier != null)
            return aggregatingClassifier.supportsWeightedData();
        else 
            return aggregatingRegressor.supportsWeightedData();
    }

    @Override
    public double regress(DataPoint data)
    {
        return aggregatingRegressor.regress(getPredVecR(data, 1.0));
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        BaseUpdateableRegressor.trainEpochs(dataSet, this, 1);
    }

    @Override
    public UpdatableStacking clone()
    {
        return new UpdatableStacking(this);
    }
    
}
