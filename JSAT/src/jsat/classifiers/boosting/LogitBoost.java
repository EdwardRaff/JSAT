
package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.OneVSAll;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;

/**
 * An implementation of the original 2 class LogitBoost algorithm. While there is a 
 * multi-class description in the original paper, its implementation is congruent 
 * with the result of using LogitBoost with {@link OneVSAll} classifier. <br>
 * LogitBoost differs from its predecessors in that it boosts 
 * {@link Regressor regression} models to create a powerful classifier. 
 * <br><br>
 * Paper: <b>Special Invited Paper Additive Logistic Regression: A Statistical
 * View of Boosting</b>, By Jerome Friedman, Trevor Hastie and Robert Tibshirani.
 * <i>The Annals of Statistics</i> 2000, Vol. 28, No. 2, 337–407
 * 
 * @author Edward Raff
 */
public class LogitBoost implements Classifier, Parameterized
{

    private static final long serialVersionUID = 1621062168467402062L;
    /**
     * The constant factor that the sum of regressors is scaled by. 
     */
    protected double fScaleConstant = 0.5;
    /**
     * Weak learners
     */
    protected List<Regressor> baseLearners;
    /**
     * Weak learner to use, 'the oracle' 
     */
    protected Regressor baseLearner;
    private int maxIterations;
    /**
     * Constant for stability and controls the maximum penalty  
     */
    private double zMax = 3;

    /**
     * Creates a new LogitBoost using the standard {@link MultipleLinearRegression} .
     * @param M the maximum number of iterations. 
     */
    public LogitBoost(int M)
    {
        this(new MultipleLinearRegression(true), M);
    }
    
    /**
     * Creates a new LogitBoost using the given base learner. 
     * @param baseLearner the weak learner to build an ensemble out of. 
     * @param M the maximum number of iterations. 
     */
    public LogitBoost(Regressor baseLearner, int M)
    {
        if(!baseLearner.supportsWeightedData())
            throw new RuntimeException("Base Learner must support weighted data points to be boosted");
        this.baseLearner = baseLearner;
        this.maxIterations = M;
    }
    
    /**
     * 
     * @return a list of the models that are in this ensemble. 
     */
    public List<Regressor> getModels()
    {
        return Collections.unmodifiableList(baseLearners);
    }

    /**
     * Sets the maximum number of iterations of boosting that can occur, giving 
     * the maximum number of base learners that may be trained
     * @param maxIterations the maximum number of iterations
     */
    public void setMaxIterations(int maxIterations)
    {
        this.maxIterations = maxIterations;
    }

    /**
     * The maximum number of iterations of boosting that may occur.
     * @return maximum number of iterations of boosting that may occur.
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets the penalty bound for miss-classification of results. This also provides
     * numerical stability to the algorithm. The results are not sensitive to this 
     * value. The recommended value range is in [2, 4]
     * 
     * @param zMax the penalty bound
     * @throws ArithmeticException if the value is not in (0, {@link Double#MAX_VALUE}]
     */
    public void setzMax(double zMax)
    {
        if(Double.isInfinite(zMax) || Double.isNaN(zMax) || zMax <= 0)
            throw new ArithmeticException("Invalid penalty given: " + zMax);
        this.zMax = zMax;
    }

    /**
     * Returns the maximum miss-classification penalty used by the algorithm. 
     * @return the maximum miss-classification
     */
    public double getzMax()
    {
        return zMax;
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        if(baseLearner == null)
            throw new UntrainedModelException("Model has not yet been trained");
        double p = P(data);
        
        CategoricalResults cr  = new CategoricalResults(2);
        
        cr.setProb(1, p);
        cr.setProb(0, 1.0-p);
        
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("LogitBoost only supports binary decision tasks, not " + dataSet.getClassSize() + " class problems");
        /**
         * The data points paired with what we will use to store the target regression values. 
         */
        List<DataPointPair<Double>> dataPoints = dataSet.getAsFloatDPPList();
        
        baseLearners = new ArrayList<Regressor>(maxIterations);
        int N = dataSet.getSampleSize();
        
        for(int m = 0; m < maxIterations; m++)
        {
            for(int i = 0; i < N; i++)
            {
                DataPoint dp = dataPoints.get(i).getDataPoint();
                double pi = P(dp);
                double zi;
                if(dataSet.getDataPointCategory(i) == 1)
                    zi = Math.min(zMax, 1.0/pi);
                else
                    zi = Math.max(-zMax, -1.0/(1.0-pi));
                double wi = Math.max(pi*(1-pi), 2*1e-15);

                dp.setWeight(wi);
                dataPoints.get(i).setPair(zi);
            }
            
            Regressor f = baseLearner.clone();
            f.train(new RegressionDataSet(dataPoints));
            baseLearners.add(f);
        }
        
    }
    
    private double F(DataPoint x)
    {
        double fx = 0.0;//0 so when we are uninitalized P will return 0.5
        
        for(Regressor fm : baseLearners)
            fx += fm.regress(x);
        return fx*fScaleConstant;
    }
    
    /**
     * Returns the probability that a given data point belongs to class 1 
     * @param x the data point in question
     * @return P(y = 1 | x)
     */
    protected double P(DataPoint x)
    {
        /**
         *              F(x)
         *             e
         * p(x) = ---------------
         *         F(x)    - F(x)
         *        e     + e
         */
        double fx = F(x);
        double efx = Math.exp(fx);
        double enfx = Math.exp(-fx);
        if(Double.isInfinite(efx) && efx > 0 && enfx < 1e-15)//Well classified point could return a Infinity which turns into NaN
            return 1.0;
        return efx/(efx + enfx);
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public LogitBoost clone()
    {
        LogitBoost clone = new LogitBoost(maxIterations);
        clone.zMax = this.zMax;
        if(this.baseLearner != null) 
            clone.baseLearner = this.baseLearner.clone();
        if(this.baseLearners != null)
        {
            clone.baseLearners = new ArrayList<Regressor>(this.baseLearners.size());
            for(Regressor r :  baseLearners)
                clone.baseLearners.add(r.clone());
        }
        return clone;
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
