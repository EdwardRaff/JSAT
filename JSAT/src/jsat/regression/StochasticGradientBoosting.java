
package jsat.regression;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.math.rootfinding.RootFinder;
import jsat.math.rootfinding.Zeroin;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * An implementation of Stochastic Gradient Boosting (SGB) for the Squared Error
 * loss. SGB is also known as Gradient Boosting Machine. There is a specialized 
 * version of SGB known as TreeBoost, that is not implemented by this method. 
 * SGB is a boosting method derived for regression. It uses many weak learners 
 * by attempting to estimate the residual error of all previous learners. It can
 * also use an initial strong learner and use the weak learners to refine the 
 * initial estimate. 
 * 
 * <br><br>
 * See papers:<br>
 * Friedman, J. H. (2002). 
 * <a href="http://onlinelibrary.wiley.com/doi/10.1002/cbdv.200490137/abstract">
 * Stochastic gradient boosting</a>. Computational Statistics&amp;Data Analysis, 
 * 38(4), 367–378. 
 * <br><br>
 * Mohan, A., Chen, Z.,&amp;Weinberger, K. (2011). 
 * <a href="http://www1.cse.wustl.edu/~kilian/papers/mohan11a.pdf">Web-search 
 * ranking with initialized gradient boosted regression trees</a>. 
 * Journal of Machine Learning Research, 14, 
 * 
 * 
 * 
 * @author Edward Raff
 */
public class StochasticGradientBoosting implements Regressor, Parameterized
{

	private static final long serialVersionUID = -2855154397476855293L;

	/**
     * The default value for the 
     * {@link #setTrainingProportion(double) training proportion} is 
     * {@value #DEFAULT_TRAINING_PROPORTION}. 
     */
    public static final double DEFAULT_TRAINING_PROPORTION = 0.5;
    
    /**
     * The default value for the {@link #setLearningRate(double) } is 
     * {@value #DEFAULT_LEARNING_RATE}
     */
    public static final double DEFAULT_LEARNING_RATE = 0.1;
    
    
    
    /**
     * The proportion of the data set to be used for each iteration of training.
     * The points that make up the iteration are a random sampling without 
     * replacement. 
     */
    private double trainingProportion;
    
    private Regressor weakLearner;
    
    private Regressor strongLearner;
    
    /**
     * The ordered list of weak learners
     */
    private List<Regressor> F;
    /**
     * The list of learner coefficients for each weak learner. 
     */
    private List<Double> coef;
    
    private double learningRate;
    
    private int maxIterations;

    /**
     * Creates a new initialized SGB learner.
     *
     * @param strongLearner the powerful learner to refine with weak learners
     * @param weakLearner the weak learner to fit to the residuals in each iteration
     * @param maxIterations the maximum number of algorithm iterations to perform
     * @param learningRate the multiplier to apply to the weak learners
     * @param trainingPortion the proportion of the data set to use for each iteration of learning
     */
    
    public StochasticGradientBoosting(Regressor strongLearner, Regressor weakLearner, int maxIterations, double learningRate, double trainingPortion)
    {
        this.trainingProportion = trainingPortion;
        this.strongLearner = strongLearner;
        this.weakLearner = weakLearner;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
    }
    
    /**
     * Creates a new SGB learner that is initialized using the weak learner.
     *
     * @param weakLearner the weak learner to fit to the residuals in each iteration
     * @param maxIterations the maximum number of algorithm iterations to perform
     * @param learningRate the multiplier to apply to the weak learners
     * @param trainingPortion the proportion of the data set to use for each iteration of learning
     */
    
    public StochasticGradientBoosting(Regressor weakLearner, int maxIterations, double learningRate, double trainingPortion)
    {
        this(null, weakLearner, maxIterations, learningRate, trainingPortion);
    }
    
    /**
     * Creates a new SGB learner that is initialized using the weak learner.
     * 
     * @param weakLearner the weak learner to fit to the residuals in each iteration
     * @param maxIterations the maximum number of algorithm iterations to perform
     * @param learningRate the multiplier to apply to the weak learners
     */
    public StochasticGradientBoosting(Regressor weakLearner, int maxIterations, double learningRate)
    {
        this(weakLearner, maxIterations, learningRate, DEFAULT_TRAINING_PROPORTION);
    }
    
    /**
     * Creates a new SGB learner that is initialized using the weak learner.
     * 
     * @param weakLearner the weak learner to fit to the residuals in each iteration
     * @param maxIterations the maximum number of algorithm iterations to perform
     */
    public StochasticGradientBoosting(Regressor weakLearner, int maxIterations)
    {
        this(weakLearner, maxIterations, DEFAULT_LEARNING_RATE);
    }

    /**
     * Sets the maximum number of iterations used in SGB. 
     * 
     * @param maxIterations the maximum number of algorithm iterations to perform
     */
    public void setMaxIterations(int maxIterations)
    {
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the maximum number of iterations used in SGB
     * @return the maximum number of algorithm iterations to perform
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets the learning rate of the algorithm. The GB version uses a learning 
     * rate of 1. SGB uses a learning rate in (0,1) to avoid overfitting. The 
     * learning rate is multiplied by the output of each weak learner to reduce 
     * its contribution. 
     * 
     * @param learningRate the multiplier to apply to the weak learners
     * @throws ArithmeticException if the learning rate is not in the range (0, 1]
     */
    public void setLearningRate(double learningRate)
    {
        //+- Inf case captured in >1 <= 0 case
        if(learningRate > 1 || learningRate <= 0 || Double.isNaN(learningRate))
            throw new ArithmeticException("Invalid learning rate");
        this.learningRate = learningRate;
    }

    /**
     * Returns the learning rate of the algorithm used to control overfitting. 
     * @return the learning rate multiplier applied to the weak learner outputs
     */
    public double getLearningRate()
    {
        return learningRate;
    }

    /**
     * The GB version uses the whole data set at each iteration. SGB can use a 
     * fraction of the data set at each iteration in order to reduce overfitting
     * and add randomness. 
     * 
     * @param trainingProportion the fraction of training the data set to use 
     * for each iteration of SGB
     * @throws ArithmeticException if the trainingPortion is not a valid 
     * fraction in (0, 1]
     */
    public void setTrainingProportion(double trainingProportion)
    {
        //+- Inf case captured in >1 <= 0 case
        if(trainingProportion > 1 || trainingProportion <= 0 || Double.isNaN(trainingProportion))
            throw new ArithmeticException("Training Proportion is invalid");
        this.trainingProportion = trainingProportion;
    }

    /**
     * Returns the fraction of the data points used during each iteration of the
     * training algorithm.  
     * 
     * @return  the fraction of the training data set to use for each 
     * iteration of SGB
     */
    public double getTrainingProportion()
    {
        return trainingProportion;
    }

    @Override
    public double regress(DataPoint data)
    {
        if(F == null || F.isEmpty())
            throw new UntrainedModelException();
        
        double result = 0;
        for(int i =0; i < F.size(); i++)
            result += F.get(i).regress(data)*coef.get(i);
        
        return result;
    }
    
    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        //use getAsDPPList to get coppies of the data points, so we can safely alter this set
        final List<DataPointPair<Double>> backingResidsList = dataSet.getAsDPPList();
        
        F = new ArrayList<Regressor>(maxIterations);
        coef = new DoubleList(maxIterations);
        
        //Add the first learner. Either an instance of the weak learner, or a strong initial estimate
        Regressor lastF = strongLearner == null ? weakLearner.clone() : strongLearner.clone();
        if(threadPool == null || threadPool instanceof FakeExecutor)
            lastF.train(dataSet);
        else
            lastF.train(dataSet, threadPool);
        F.add(lastF);
        coef.add(learningRate*getMinimizingErrorConst(backingResidsList, lastF));
        
        /**
         * Instead of recomputing previous weak learner's output, keep track of 
         * the current total sum to know the current prediction value
         */
        final double[] currPredictions = new double[dataSet.getSampleSize()];
        
        
        
        /**
         * The residuals
         */
        RegressionDataSet resids = RegressionDataSet.usingDPPList(backingResidsList);
        
        
        final int randSampleSize = (int) Math.round(resids.getSampleSize()*trainingProportion);
        final List<DataPointPair<Double>> randSampleList = new ArrayList<DataPointPair<Double>>(randSampleSize);
        final Random rand = RandomUtil.getRandom();

        for(int iter = 0; iter < maxIterations; iter++)
        {
            final double lastCoef = coef.get(iter);
            lastF = F.get(iter);
            
            //Compute the new residuals 
            for(int j = 0; j < resids.getSampleSize(); j++)
            {
                //Update the current total preduction values while we do this 
                double lastFPred = lastF.regress(resids.getDataPoint(j));
                currPredictions[j] += lastCoef*lastFPred;
                
                //The next set of residuals could be computed from the previous,
                //but its more stable to just take the total residuals fromt he 
                //source each time
                resids.setTargetValue(j, dataSet.getTargetValue(j)-currPredictions[j]);
            }
            
            
            
            //Take a random sample
            randSampleList.clear();
            ListUtils.randomSample(backingResidsList, randSampleList, randSampleSize, rand);
            
            final Regressor h = weakLearner.clone();
            final RegressionDataSet tmpDataSet = RegressionDataSet.usingDPPList(randSampleList);
            
            if(threadPool == null || threadPool instanceof FakeExecutor)
                h.train(tmpDataSet);
            else
                h.train(tmpDataSet, threadPool);
            
            
            double y = getMinimizingErrorConst( backingResidsList, h);
            
            F.add(h);
            coef.add(learningRate*y);
        }
    }
    
    /**
     * Finds the constant <tt>y</tt> such that the squared error of the 
     * Regressor <tt>h</tt> on the set of residuals <tt>backingResidsList</tt> 
     * is minimized. 
     * @param backingResidsList the DataPointPair list of residuals
     * @param h the regressor that is having the error of its output minimized
     * @return the constant <tt>y</tt> that minimizes the squared error of the regressor on the training set. 
     */
    private double getMinimizingErrorConst(final List<DataPointPair<Double>> backingResidsList, final Regressor h)
    {
        //Find the coeficent that minimized the residual error by finding the zero of its derivative (local minima)
        Function fhPrime = getDerivativeFunc(backingResidsList, h);
        RootFinder rf = new Zeroin();
        double y = rf.root(1e-4, 50, new double[]{-2.5, 2.5}, fhPrime, 0, 1.0);
        return y;
    }
    
    /**
     * Returns a function object that approximates the derivative of the squared
     * error of the Regressor as a function of the constant factor multiplied on
     * the Regressor's output. 
     * 
     * @param backingResidsList the DataPointPair list of residuals
     * @param h the regressor that is having the error of its output minimized
     * @return a Function object approximating the derivative of the squared error
     */
    private Function getDerivativeFunc(final List<DataPointPair<Double>> backingResidsList, final Regressor h)
    {
        final FunctionBase fhPrime = new FunctionBase()
        {
            /**
			 * 
			 */
			private static final long serialVersionUID = -2211642040228795719L;

			@Override
            public double f(Vec x)
            {
                double c1 = x.get(0);//c2=c1-eps
                double eps = 1e-5;
                double c1Pc2 = c1 * 2 - eps;//c1+c2 = c1+c1-eps
                double result = 0;
                /*
                 * Computing the estimate of the derivative directly, f'(x) approx = f(x)-f(x-eps)
                 * 
                 * hEst is the output of the new regressor, target is the true residual target value
                 * 
                 * So we have several 
                 * (hEst_i   c1 - target)^2 - (hEst_i   c2 -target)^2   //4 muls, 3 subs
                 * Where c2 = c1-eps
                 * Which simplifies to
                 * (c1 - c2) hEst ((c1 + c2) hEst - 2 target)
                 * =
                 * eps hEst (c1Pc2 hEst - 2 target)//3 muls, 1 sub, 1 shift (mul by 2) 
                 * 
                 * because eps is on the outside and independent of each 
                 * individual summation, we can move it out and do the eps
                 * multiplicatio ont he final result.  Reducing us to 
                 * 
                 * 2 muls, 1 sub, 1 shift (mul by 2)
                 * 
                 * per loop
                 * 
                 * Which reduce computation, and allows us to get the result
                 * in one pass of the data
                 */

                for (DataPointPair<Double> dpp : backingResidsList)
                {
                    double hEst = h.regress(dpp.getDataPoint());
                    double target = dpp.getPair();

                    result += hEst * (c1Pc2 * hEst - 2 * target);
                }

                return result * eps;
            }
        };

        return fhPrime;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        if(strongLearner != null)
            return strongLearner.supportsWeightedData() && weakLearner.supportsWeightedData();
        
        return weakLearner.supportsWeightedData();
    }

    @Override
    public StochasticGradientBoosting clone()
    {
        StochasticGradientBoosting clone = new StochasticGradientBoosting(weakLearner.clone(), maxIterations, learningRate, trainingProportion);
        
        if(F != null)
        {
            clone.F = new ArrayList<Regressor>(F.size());
            for(Regressor f : this.F)
                clone.F.add(f.clone());
        }
        if(coef != null)
        {
            clone.coef = new DoubleList(this.coef.size());
            for(double d : this.coef)
                clone.coef.add(d);
        }
        
        if(strongLearner != null)
            clone.strongLearner = this.strongLearner.clone();
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
