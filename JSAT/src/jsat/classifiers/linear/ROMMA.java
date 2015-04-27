package jsat.classifiers.linear;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Provides an implementation of the linear Relaxed online Maximum Margin 
 * algorithm, which finds a similar solution to SVMs. By default, the aggressive
 * variant with an implicit bias term is used, which is the suggested form from 
 * the paper. It is a binary classifier. 
 * <br><br>
 * See: Li, Y.,&amp;Long, P. M. (2002). <i>The Relaxed Online Maximum Margin 
 * Algorithm</i>. Machine Learning, 46(1-3), 361–387. 
 * doi:10.1023/A:1012435301888
 * 
 * @author Edward Raff
 */
public class ROMMA extends BaseUpdateableClassifier implements BinaryScoreClassifier, SingleWeightVectorModel
{

	private static final long serialVersionUID = 8163937542627337711L;
	private boolean useBias = true;
    private boolean aggressive;
    private Vec w;
    private double bias;

    /**
     * Creates a new aggressive ROMMA classifier
     */
    public ROMMA()
    {
        this(true);
    }

    /**
     * Creates a new ROMMA classifier
     * @param aggressive whether or not to use the aggressive variant
     */
    public ROMMA(boolean aggressive)
    {
        setAggressive(aggressive);
    }

    /**
     * Copy constructor
     * @param other the ROMMA object to copy
     */
    protected ROMMA(ROMMA other)
    {
        this.aggressive = other.aggressive;
        if(other.w != null)
            this.w = other.w;
        this.bias = other.bias;
        this.useBias = other.useBias;
    }
    

    @Override
    public ROMMA clone()
    {
        return new ROMMA(this);
    }

    /**
     * Determines whether the normal or aggressive ROMMA algorithm will be used. 
     * @param aggressive {@code true} to use the aggressive variant
     */
    public void setAggressive(boolean aggressive)
    {
        this.aggressive = aggressive;
    }

    /**
     * Returns whether or not the aggressive variant of ROMMA is used
     * @return {@code true} if the aggressive variant of ROMMA is used
     */
    public boolean isAggressive()
    {
        return aggressive;
    }

    /**
     * Sets whether or not an implicit bias term will be added to the data set
     * @param useBias {@code true} to add an implicit bias term
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns whether or not an implicit bias term is in use
     * @return {@code true} if a bias term is in use
     */
    public boolean isUseBias()
    {
        return useBias;
    }
    
    /**
     * Returns the weight vector used to compute results via a dot product. <br>
     * Do not modify this value, or you will alter the results returned.
     * @return the learned weight vector for prediction
     */
    public Vec getWeightVec()
    {
        return w;
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
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes <= 0)
            throw new FailedToFitException("ROMMA requires numerical features");
        else if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("ROMMA only supports binary classification");
        w = new DenseVector(numericAttributes);
        bias = 0;
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        Vec x = dataPoint.getNumericalValues();
        double wx = w.dot(x)+bias;
        double y = targetClass*2-1;
        double pred = y*wx;
        if(pred < 1)
        {
            final double ww = w.dot(w);
            final double xx = x.dot(x);
            final double wwxx = ww*xx;
            if(aggressive)
            {
                if(pred >= wwxx)
                {
                    w.zeroOut();
                    w.mutableAdd(y/xx, x);
                    if(useBias)
                        bias = y/xx;
                    return;
                }
            }
            //else / normal
            double denom = wwxx - wx * wx;
            double c = (wwxx - pred) / denom;
            double d = (ww * (y - wx)) / denom;
            w.mutableMultiply(c);
            w.mutableAdd(d, x);
            if(useBias)
                bias = c*bias + d;
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(w == null)
            throw new UntrainedModelException("Model has not been trained");
        double wx = getScore(data);
        CategoricalResults cr = new CategoricalResults(2);
        if(wx < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return w.dot(dp.getNumericalValues())+bias;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
}
