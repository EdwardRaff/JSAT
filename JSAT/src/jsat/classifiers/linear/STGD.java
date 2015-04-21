package jsat.classifiers.linear;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.BaseUpdateableRegressor;
import jsat.regression.RegressionDataSet;
import jsat.regression.UpdateableRegressor;

/**
 * This provides an implementation of Sparse Truncated Gradient Descent for 
 * L<sub>1</sub> regularized linear classification and regression on sparse data
 * sets. 
 * <br><br>
 * Unlike normal L<sub>1</sub> regression, regularization is controlled by the 
 * {@link #setGravity(double) gravity} parameter, but other parameters 
 * contribute to the level of sparsity. 
 * <br><br>
 * See: Langford, J., Li, L.,&amp;Zhang, T. (2009). <i>Sparse online learning via 
 * truncated gradient</i>. The Journal of Machine Learning Research, 10, 
 * 777–801. Retrieved from <a href="http://dl.acm.org/citation.cfm?id=1577097">
 * here</a>
 * @author Edward Raff
 */
public class STGD extends BaseUpdateableClassifier implements UpdateableRegressor, BinaryScoreClassifier, Parameterized, SingleWeightVectorModel
{

	private static final long serialVersionUID = 5753298014967370769L;

	private Vec w;
    
    private int K;
    private double learningRate;
    private double threshold;
    private double gravity;
    
    private int time;
    private int[] t;
    
    /**
     * Creates a new STGD learner
     * @param K the regularization frequency
     * @param learningRate the learning rate to use
     * @param threshold the regularization threshold
     * @param gravity the regularization parameter
     */
    public STGD(int K, double learningRate, double threshold, double gravity)
    {
        setK(K);
        setLearningRate(learningRate);
        setThreshold(threshold);
        setGravity(gravity);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected STGD(STGD toCopy)
    {
        if(toCopy.w != null)
            this.w = toCopy.w.clone();
        this.K = toCopy.K;
        this.learningRate = toCopy.learningRate;
        this.threshold = toCopy.threshold;
        this.gravity = toCopy.gravity;
        this.time = toCopy.time;
        if(toCopy.t != null)
            this.t = Arrays.copyOf(toCopy.t, toCopy.t.length);
    }

    /**
     * Sets the frequency of applying the {@link #setGravity(double) gravity} 
     * parameter to the weight vector. This value must be positive, and the 
     * gravity will be applied every <i>K</i> updates.  Increasing this value 
     * encourages greater sparsity. 
     * 
     * @param K the frequency to apply regularization in [1, Infinity )
     */
    public void setK(int K)
    {
        if(K < 1)
            throw new IllegalArgumentException("K must be positive, not " + K);
        this.K = K;
    }

    /**
     * Returns the frequency of regularization application
     * @return the frequency of regularization application
     */
    public int getK()
    {
        return K;
    }
    
    /**
     * Sets the learning rate to use
     * @param learningRate the learning rate &gt; 0. 
     */
    public void setLearningRate(double learningRate)
    {
        if(Double.isInfinite(learningRate) || Double.isNaN(learningRate) || learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive, not " + learningRate);
        this.learningRate = learningRate;
    }

    /**
     * Returns the learning rate to use
     * @return  the learning rate to use
     */
    public double getLearningRate()
    {
        return learningRate;
    }

    /**
     * Sets the threshold for a coefficient value to avoid regularization. While
     * a coefficient reaches this magnitude, regularization will not be applied. 
     * @param threshold the coefficient regularization threshold in 
     * ( 0, Infinity ]
     */
    public void setThreshold(double threshold)
    {
        if(Double.isNaN(threshold) || threshold <= 0)
            throw new IllegalArgumentException("Threshold must be positive, not " + threshold);
        this.threshold = threshold;
    }

    /**
     * Returns the coefficient threshold parameter
     * @return the coefficient threshold parameter
     */
    public double getThreshold()
    {
        return threshold;
    }

    /**
     * Sets the gravity regularization parameter that "weighs down" the
     * coefficient values. Larger gravity values impose stronger regularization,
     * and encourage greater sparsity. 
     * 
     * @param gravity the regularization parameter in ( 0, Infinity )
     */
    public void setGravity(double gravity)
    {
        if(Double.isInfinite(gravity) || Double.isNaN(gravity) || gravity <= 0)
            throw new IllegalArgumentException("Gravity must be positive, not " + gravity);
        this.gravity = gravity;
    }

    /**
     * Returns the regularization parameter
     * @return the regularization parameter
     */
    public double getGravity()
    {
        return gravity;
    }

    @Override
    public Vec getRawWeight()
    {
        return w;
    }

    @Override
    public double getBias()
    {
        return 0;
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
    public STGD clone()
    {
        return new STGD(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("STGD supports only binary classification");
        setUp(categoricalAttributes, numericAttributes);
    }
    
    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes)
    {
        if(numericAttributes < 1)
            throw new FailedToFitException("STGD requires numeric features");
        w = new DenseVector(numericAttributes);
        t = new int[numericAttributes];
    }
    
    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        BaseUpdateableRegressor.trainEpochs(dataSet, this, getEpochs());
    }
    
    private static double T(double v_j, double a, double theta)
    {
        if(v_j >= 0 && v_j <= theta)
            return Math.max(0, v_j-a);
        else if(v_j <= 0 && v_j >= -theta)
            return Math.min(0, v_j+a);
        else
            return v_j;
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        time++;
        final Vec x = dataPoint.getNumericalValues();
        final int y = targetClass*2-1;
        final int yHat = (int) Math.signum(w.dot(x));
        if(yHat == y)//Not part of the described algorithm (using signum), but needed
            return;
        performUpdate(x, y, yHat);
    }
    
    @Override
    public void update(DataPoint dataPoint, final double y)
    {
        time++;
        final Vec x = dataPoint.getNumericalValues();
        final double yHat = w.dot(x);
        performUpdate(x, y, yHat);
    }
    
    /**
     * Performs the sparse update of the weight vector
     * @param x the input vector
     * @param y the true value
     * @param yHat the predicted value
     */
    private void performUpdate(final Vec x, final double y, final double yHat)
    {
        for(IndexValue iv : x)
        {
            final int j = iv.getIndex();
            w.set(j, 
                    T(w.get(j)+2*learningRate*(y-yHat)*iv.getValue(), 
                    ((time-t[j])/K)*gravity*learningRate,
                    threshold));
            
            t[j] += ((time-t[j])/K)*K;
        }
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        if(getScore(data) > 0)
            cr.setProb(1, 1.0);
        else
            cr.setProb(0, 1.0);
        return cr;
    }
    
    @Override
    public double regress(DataPoint data)
    {
        return getScore(data);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return w.dot(dp.getNumericalValues());
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
