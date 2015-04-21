
package jsat.regression;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.NoDecay;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * A Stochastic implementation of Ridge Regression. Ridge 
 * Regression is equivalent to {@link MultipleLinearRegression} with an added 
 * L<sub>2</sub> penalty for the weight vector. <br><br>
 * This algorithm works best for problems with a large number of data points or
 * very high dimensional problems. 
 * 
 * @author Edward Raff
 */
public class StochasticRidgeRegression implements Regressor, Parameterized, SingleWeightVectorModel
{

	private static final long serialVersionUID = -3462783438115627128L;
	private double lambda;
    private int epochs;
    private int batchSize;
    private double learningRate;
    private DecayRate learningDecay;
    private Vec w;
    private double bias;

    /**
     * Creates a new stochastic Ridge Regression learner that does not use a 
     * decay rate
     * @param lambda the regularization term 
     * @param epochs the number of training epochs to perform
     * @param batchSize the batch size for updates
     * @param learningRate the learning rate 
     */
    public StochasticRidgeRegression(double lambda, int epochs, int batchSize, double learningRate)
    {
        this(lambda, epochs, batchSize, learningRate, new NoDecay());
    }
    
    /**
     * Creates a new stochastic Ridge Regression learner
     * @param lambda the regularization term 
     * @param epochs the number of training epochs to perform
     * @param batchSize the batch size for updates
     * @param learningRate the learning rate 
     * @param learningDecay the learning rate decay
     */
    public StochasticRidgeRegression(double lambda, int epochs, int batchSize, double learningRate, DecayRate learningDecay)
    {
        setLambda(lambda);
        setEpochs(epochs);
        setBatchSize(batchSize);
        setLearningRate(learningRate);
        setLearningDecay(learningDecay);
    }
    
    /**
     * Sets the regularization parameter used.  
     * @param lambda the positive regularization constant in (0, Inf)
     */
    public void setLambda(double lambda)
    {
        if(Double.isNaN(lambda) || Double.isInfinite(lambda) || lambda <= 0)
            throw new IllegalArgumentException("lambda must be a positive constant, not " + lambda);
        this.lambda = lambda;
    }

    /**
     * Returns the regularization constant in use
     * @return the regularization constant in use 
     */
    public double getLambda()
    {
        return lambda;
    }

    /**
     * Sets the learning rate used, and should be in the range (0, 1). 
     * 
     * @param learningRate the learning rate to use
     */
    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    /**
     * Returns the learning rate in use. 
     * @return the learning rate to use. 
     */
    public double getLearningRate()
    {
        return learningRate;
    }

    /**
     * Sets the learning rate decay function to use. The decay is applied after 
     * each epoch through the data set. Using a decay rate can reduce the time 
     * to converge and quality of the solution for difficult problems. 
     * @param learningDecay the decay function to apply to the learning rate
     */
    public void setLearningDecay(DecayRate learningDecay)
    {
        this.learningDecay = learningDecay;
    }

    /**
     * Returns the learning decay rate used
     * @return the learning decay rate used
     */
    public DecayRate getLearningDecay()
    {
        return learningDecay;
    }

    /**
     * Sets the batch size to learn from. If larger than the training set, the 
     * problem will reduce to classic gradient descent. 
     * 
     * @param batchSize the number of training points to use in each batch update
     */
    public void setBatchSize(int batchSize)
    {
        if(batchSize <= 0)
            throw new IllegalArgumentException("Batch size must be a positive constant, not " + batchSize);
        this.batchSize = batchSize;
    }

    /**
     * Returns the batch size for updates
     * @return the batch size for updates
     */
    public int getBatchSize()
    {
        return batchSize;
    }

    /**
     * Sets the number of iterations through the whole training set that will be
     * performed. 
     * @param epochs the number of training iterations 
     */
    public void setEpochs(int epochs)
    {
        if(epochs <= 0)
            throw new IllegalArgumentException("At least one epoch must be performed, can not use " + epochs);
        this.epochs = epochs;
    }

    /**
     * Returns the number of training iterations
     * @return the number of training iterations
     */
    public int getEpochs()
    {
        return epochs;
    }

    @Override
    public Vec getRawWeight()
    {
        return w;
    }

    @Override
    public double getBias()
    {
        return bias;
    }
    
    @Override
    public Vec getRawWeight(int index)
    {
        if(index < 1)
            return getRawWeight();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public double getBias(int index)
    {
        if (index < 1)
            return getBias();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public int numWeightsVecs()
    {
        return 1;
    }

    @Override
    public double regress(DataPoint data)
    {
        return regress(data.getNumericalValues());
    }
    
    private double regress(Vec data)
    {
        return w.dot(data) + bias;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        int batch = Math.min(batchSize, dataSet.getSampleSize());
        w = new DenseVector(dataSet.getNumNumericalVars());
        
        IntList sample = new IntList(dataSet.getSampleSize());
        ListUtils.addRange(sample, 0, dataSet.getSampleSize(), 1);
        
        //Time and last time used to lazy update the parameters that do not get touched on a sparse update
        int time = 0;
        
        double[] errors = new double[batch];
        
        final boolean sparseUpdates;
        {
            int sparse = 0;
            for (int i = 0; i < dataSet.getSampleSize(); i++)
                if(dataSet.getDataPoint(i).getNumericalValues().isSparse())
                    sparse++;
            if(sparse > dataSet.getSampleSize()/4)
                sparseUpdates = true;
            else
                sparseUpdates = false;
        }
        
        int[] lastTime = sparseUpdates ? new int[w.length()] : null;
        
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            Collections.shuffle(sample);
            
            final double alpha = learningDecay.rate(epoch, epochs, learningRate)/batch;
            final double alphaReg = alpha*lambda;
            
            for(int i = 0; i < sample.size(); i+= batch)
            {
                if(i+batch >= sample.size())
                    continue;//skip, not enough in the batch
                
                time++;
                //get errors
                for(int b = i; b < i+batch; b++)
                    errors[b-i] = regress(dataSet.getDataPoint(sample.get(i)))-dataSet.getTargetValue(sample.get(i));
                
                //perform updates 
                for(int b = i; b < i+batch; b++)
                {
                    final double error = errors[b-i];
                    final double alphaError = alpha*error;
                    //update bias
                    bias -= alphaError;
                    Vec x = dataSet.getDataPoint(sample.get(i)).getNumericalValues();
                    
                    if(sparseUpdates)
                    {
                        for(IndexValue iv : x)
                        {
                            int idx = iv.getIndex();
                            if(lastTime[idx] != time)//update the theta for all missed updates
                            {
                                double theta_idx = w.get(idx);
                                w.set(idx, theta_idx*Math.pow(1-alphaReg, time-lastTime[idx]));
                                lastTime[idx] = time;
                            }
                            //now accumlate errors
                            w.increment(idx, -alphaError*iv.getValue());
                        }
                    }
                    else//dense updates, no need to track last time we updated weight values
                    {
                        if(b == i)//update on first access
                            w.mutableMultiply(1-alphaReg);
                        //add error
                        w.mutableSubtract(alphaError, x);
                    }
                }
            }
            
            /*
             * if sparse, accumulate missing weight updates due to 
             * regularization. If the learning rate changes, the weights must be
             * updated at the end of every epoch. If the learing rate is 
             * constant, we only have to update on the last epoch
             */
            if (sparseUpdates && ( !(learningDecay instanceof NoDecay) || (epoch == epochs-1) ))
            {
                for (int idx = 0; idx < w.length(); idx++)
                {
                    if (lastTime[idx] != time)//update the theta for all missed updates
                    {
                        double theta_idx = w.get(idx);
                        w.set(idx, theta_idx * Math.pow(1 - alphaReg, time - lastTime[idx]));
                        lastTime[idx] = time;
                    }
                }
            }
        }
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public StochasticRidgeRegression clone()
    {
        StochasticRidgeRegression clone = new StochasticRidgeRegression(lambda, epochs, batchSize, learningRate, learningDecay);
        if(this.w != null)
            clone.w = this.w.clone();
        clone.bias = this.bias;
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
