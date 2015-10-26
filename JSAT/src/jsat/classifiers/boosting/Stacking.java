
package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.classifiers.linear.LinearBatch;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;

/**
 * This provides an implementation of the Stacking ensemble method. Stacking 
 * learns several base classifiers and a top level classifier learns to predict 
 * the target based on the outputs of all the ensambled models. Historically a 
 * linear model (such as {@link LinearBatch}) is used, which translates to 
 * learning a weighted vote of the classifier outputs. However any classifier 
 * may be used so long as it supports the desired target type. <br>
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
public class Stacking implements Classifier, Regressor
{

	private static final long serialVersionUID = -6173323872903232074L;
	private int folds;
    /**
     * The number of weights needed per model
     */
    private int weightsPerModel;
    private Classifier aggregatingClassifier;
    private List<Classifier> baseClassifiers;
    
    private Regressor aggregatingRegressor;
    private List<Regressor> baseRegressors;
    
    public static final int DEFAULT_FOLDS = 3;

    /**
     * Creates a new Stacking classifier
     * @param folds the number of cross validation folds for learning the base model
     * @param aggregatingClassifier the classifier used to merge the results of all the input classifiers
     * @param baseClassifiers the list of base classifiers to ensemble
     */
    public Stacking(final int folds, final Classifier aggregatingClassifier, final List<Classifier> baseClassifiers)
    {
        if(baseClassifiers.size() < 2) {
          throw new IllegalArgumentException("base classifiers must contain at least 2 elements, not " + baseClassifiers.size());
        }
        setFolds(folds);
        this.aggregatingClassifier = aggregatingClassifier;
        this.baseClassifiers = baseClassifiers;
        
        boolean allRegressors = aggregatingClassifier instanceof Regressor;
        for(final Classifier cl : baseClassifiers) {
          if (!(cl instanceof Regressor)) {
            allRegressors = false;
          }
        }
        
        if(allRegressors)
        {
            aggregatingRegressor = (Regressor) aggregatingClassifier;
            baseRegressors = (List) baseClassifiers;//ugly type easure exploitation... 
        }
    }
    
    /**
     * Creates a new Stacking classifier
     * @param folds the number of cross validation folds for learning the base model
     * @param aggregatingClassifier the classifier used to merge the results of all the input classifiers
     * @param baseClassifiers the array of base classifiers to ensemble
     */
    public Stacking(final int folds, final Classifier aggregatingClassifier, final Classifier... baseClassifiers)
    {
        this(folds, aggregatingClassifier, Arrays.asList(baseClassifiers));
    }
    
    /**
     * Creates a new Stacking classifier that uses {@value #DEFAULT_FOLDS} folds of cross validation
     * @param aggregatingClassifier the classifier used to merge the results of all the input classifiers
     * @param baseClassifiers the list of base classifiers to ensemble
     */
    public Stacking(final Classifier aggregatingClassifier, final List<Classifier> baseClassifiers)
    {
        this(DEFAULT_FOLDS, aggregatingClassifier, baseClassifiers);
    }
    
    /**
     * Creates a new Stacking classifier that uses {@value #DEFAULT_FOLDS} folds of cross validation
     * @param aggregatingClassifier the classifier used to merge the results of all the input classifiers
     * @param baseClassifiers the array of base classifiers to ensemble
     */
    public Stacking(final Classifier aggregatingClassifier, final Classifier... baseClassifiers)
    {
        this(DEFAULT_FOLDS, aggregatingClassifier, baseClassifiers);
    }
    
    /**
     * Creates a new Stacking regressor
     * @param folds the number of cross validation folds for learning the base model
     * @param aggregatingRegressor the regressor used to merge the results of all the input classifiers
     * @param baseRegressors the list of base regressors to ensemble
     */
    public Stacking(final int folds, final Regressor aggregatingRegressor, final List<Regressor> baseRegressors)
    {
        setFolds(folds);
        this.aggregatingRegressor = aggregatingRegressor;
        this.baseRegressors = baseRegressors;
        
        boolean allClassifiers = aggregatingRegressor instanceof Classifier;
        for(final Regressor reg : baseRegressors) {
          if (!(reg instanceof Classifier)) {
            allClassifiers = false;
          }
        }
        
        if(allClassifiers)
        {
            aggregatingClassifier = (Classifier) aggregatingRegressor;
            baseClassifiers = (List) baseRegressors;//ugly type easure exploitation... 
        }
    }

    /**
     * Creates a new Stacking regressor
     * @param folds the number of cross validation folds for learning the base model
     * @param aggregatingRegressor the regressor used to merge the results of all the input classifiers
     * @param baseRegressors the array of base regressors to ensemble
     */
    public Stacking(final int folds, final Regressor aggregatingRegressor, final Regressor... baseRegressors)
    {
        this(folds, aggregatingRegressor, Arrays.asList(baseRegressors));
    }
    
    /**
     * Creates a new Stacking regressor that uses {@value #DEFAULT_FOLDS} folds of cross validation
     * @param aggregatingRegressor the regressor used to merge the results of all the input classifiers
     * @param baseRegressors the list of base regressors to ensemble
     */
    public Stacking(final Regressor aggregatingRegressor, final List<Regressor> baseRegressors)
    {
        this(DEFAULT_FOLDS, aggregatingRegressor, baseRegressors);
    }
    
    /**
     * Creates a new Stacking regressor that uses {@value #DEFAULT_FOLDS} folds of cross validation
     * @param aggregatingRegressor the regressor used to merge the results of all the input classifiers
     * @param baseRegressors the array of base regressors to ensemble
     */
    public Stacking(final Regressor aggregatingRegressor, final Regressor... baseRegressors)
    {
        this(DEFAULT_FOLDS, aggregatingRegressor, baseRegressors);
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public Stacking(final Stacking toCopy)
    {
        this.folds = toCopy.folds;
        this.weightsPerModel = toCopy.weightsPerModel;
        if(toCopy.aggregatingClassifier != null)
        {
            this.aggregatingClassifier = toCopy.aggregatingClassifier.clone();
            this.baseClassifiers = new ArrayList<Classifier>(toCopy.baseClassifiers.size());
            for(final Classifier bc : toCopy.baseClassifiers) {
              this.baseClassifiers.add(bc.clone());
            }
            
            if(toCopy.aggregatingRegressor == toCopy.aggregatingClassifier)//supports both
            {
                aggregatingRegressor = (Regressor) aggregatingClassifier;
                baseRegressors = (List) baseClassifiers;//ugly type easure exploitation... 
            }
        }
        else//we are doing with regressors only
        {
            this.aggregatingRegressor = toCopy.aggregatingRegressor.clone();
            this.baseRegressors = new ArrayList<Regressor>(toCopy.baseRegressors.size());
            for(final Regressor br : toCopy.baseRegressors) {
              this.baseRegressors.add(br.clone());
            }
        }
    }
    
    /**
     * Sets the number of folds of cross validation to use when creating the new
     * set of weights that will be feed into the aggregating model. <br>
     * Note that the number of folds may be 1, and will run significantly 
     * faster since models do not need to be re-trained. However it will be more
     * prone to overfitting. 
     * @param folds the number of cross validation folds to use
     */
    public void setFolds(final int folds)
    {
        if(folds < 1) {
          throw new IllegalArgumentException("Folds must be a positive integer, not " + folds);
        }
        this.folds = folds;
    }

    /**
     * 
     * @return the number of CV folds used for training
     */
    public int getFolds()
    {
        return folds;
    }
    

    @Override
    public CategoricalResults classify(final DataPoint data)
    {
        final Vec w = new DenseVector(weightsPerModel*baseClassifiers.size());
        if(weightsPerModel == 1) {
          for (int i = 0; i < baseClassifiers.size(); i++) {
            w.set(i, baseClassifiers.get(i).classify(data).getProb(0)*2-1);
          }
        } else
        {
            for(int i = 0; i < baseClassifiers.size(); i++)
            {
                final CategoricalResults pred = baseClassifiers.get(i).classify(data);
                for(int j = 0; j < weightsPerModel; j++) {
                  w.set(i*weightsPerModel+j, pred.getProb(j));
            }
            }
                    
        }
        
        return aggregatingClassifier.classify(new DataPoint(w));
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
    {
        final int models = baseClassifiers.size();
        final int C = dataSet.getClassSize();
        weightsPerModel = C == 2 ? 1 : C;
        final ClassificationDataSet metaSet = new ClassificationDataSet(weightsPerModel*models, new CategoricalData[0], dataSet.getPredicting());
        
        final List<ClassificationDataSet> dataFolds = dataSet.cvSet(folds);
        //iterate in the order of the folds so we get the right dataum weights
        for(final ClassificationDataSet cds : dataFolds) {
          for (int i = 0; i < cds.getSampleSize(); i++) {
            metaSet.addDataPoint(new DenseVector(weightsPerModel*models), cds.getDataPointCategory(i), cds.getDataPoint(i).getWeight());
          }
        }
        
        //create the meta training set
        for(int c = 0; c < baseClassifiers.size(); c++)
        {
            final Classifier cl = baseClassifiers.get(c);
            int pos = 0;
            for(int f = 0; f < dataFolds.size(); f++)
            {
                final ClassificationDataSet train = ClassificationDataSet.comineAllBut(dataFolds, f);
                final ClassificationDataSet test = dataFolds.get(f);
                if(threadPool == null) {
                  cl.trainC(train);
                } else {
                  cl.trainC(train, threadPool);
                }
                for(int i = 0; i < test.getSampleSize(); i++)//evaluate and mark each point in the held out fold.
                {
                    final CategoricalResults pred  = cl.classify(test.getDataPoint(i));
                    if(C == 2) {
                      metaSet.getDataPoint(pos).getNumericalValues().set(c, pred.getProb(0)*2-1);
                    } else
                    {
                        final Vec toSet = metaSet.getDataPoint(pos).getNumericalValues();
                        for(int j = weightsPerModel*c; j < weightsPerModel*(c+1); j++) {
                          toSet.set(j, pred.getProb(j-weightsPerModel*c));
                      }
                    }
                    
                    pos++;
                }
            }
        }
        
        //train the meta model
        if(threadPool == null) {
          aggregatingClassifier.trainC(metaSet);
        } else {
          aggregatingClassifier.trainC(metaSet, threadPool);
        }
        
        //train the final classifiers, unless folds=1. In that case they are already trained
        if(folds != 1)
        {
            for(final Classifier cl : baseClassifiers) {
              if (threadPool == null) {
                cl.trainC(dataSet);
              } else {
                cl.trainC(dataSet, threadPool);
              }
            }
        }
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        if(aggregatingClassifier != null) {
          return aggregatingClassifier.supportsWeightedData();
        } else {
          return aggregatingRegressor.supportsWeightedData();
        }
    }

    @Override
    public double regress(final DataPoint data)
    {
        final Vec w = new DenseVector(baseRegressors.size());
        for (int i = 0; i < baseRegressors.size(); i++) {
          w.set(i, baseRegressors.get(i).regress(data));
        }

        return aggregatingRegressor.regress(new DataPoint(w));
    }

    @Override
    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool)
    {
        final int models = baseRegressors.size();
        weightsPerModel = 1;
        final RegressionDataSet metaSet = new RegressionDataSet(models, new CategoricalData[0]);
        
        final List<RegressionDataSet> dataFolds = dataSet.cvSet(folds);
        //iterate in the order of the folds so we get the right dataum weights
        for(final RegressionDataSet rds : dataFolds) {
          for (int i = 0; i < rds.getSampleSize(); i++) {
            metaSet.addDataPoint(new DataPoint(new DenseVector(weightsPerModel*models), rds.getDataPoint(i).getWeight()), rds.getTargetValue(i));
          }
        }
        
        //create the meta training set
        for(int c = 0; c < baseRegressors.size(); c++)
        {
            final Regressor reg = baseRegressors.get(c);
            int pos = 0;
            for(int f = 0; f < dataFolds.size(); f++)
            {
                final RegressionDataSet train = RegressionDataSet.comineAllBut(dataFolds, f);
                final RegressionDataSet test = dataFolds.get(f);
                if(threadPool == null) {
                  reg.train(train);
                } else {
                  reg.train(train, threadPool);
                }
                for(int i = 0; i < test.getSampleSize(); i++)//evaluate and mark each point in the held out fold.
                {
                    final double pred  = reg.regress(test.getDataPoint(i));
                    
                    metaSet.getDataPoint(pos++).getNumericalValues().set(c, pred);
                }
            }
        }
        
        //train the meta model
        if(threadPool == null) {
          aggregatingRegressor.train(metaSet);
        } else {
          aggregatingRegressor.train(metaSet, threadPool);
        }
        
        //train the final classifiers, unless folds=1. In that case they are already trained
        if(folds != 1)
        {
            for(final Regressor reg : baseRegressors) {
              if (threadPool == null) {
                reg.train(dataSet);
              } else {
                reg.train(dataSet, threadPool);
              }
            }
        }
    }

    @Override
    public void train(final RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public Stacking clone()
    {
        return new Stacking(this);
    }
    
}
