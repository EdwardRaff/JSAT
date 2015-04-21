package jsat.classifiers.linear;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Provides a linear implementation of the ALMAp algorithm for p = 2, which is 
 * considerably more efficient to compute. It is a binary classifier for numeric
 * features. 
 * <br>
 * ALMA requires one major parameter {@link #setAlpha(double) alpha} to be set, 
 * the other two have default behavior / values that have provable convergence.
 * <br><br>
 * See: Gentile, C. (2002). <i>A New Approximate Maximal Margin Classification 
 * Algorithm</i>. The Journal of Machine Learning Research, 2, 213–242. 
 * Retrieved from <a href="http://dl.acm.org/citation.cfm?id=944811">here</a>
 * 
 * @author Edward Raff
 */
public class ALMA2 extends BaseUpdateableClassifier implements BinaryScoreClassifier, SingleWeightVectorModel
{

	private static final long serialVersionUID = -4347891273721908507L;
	private Vec w;
    private static final double p = 2;
    private double alpha;
    private double B;
    private double C = Math.sqrt(2);
    private int k;
    
    private boolean useBias = true;
    private double bias;

    /**
     * Creates a new ALMA learner using an alpha of 0.8
     */
    public ALMA2()
    {
        this(0.8);
    }
    
    /**
     * Creates a new ALMA learner using the given alpha 
     * @param alpha the alpha value to use
     * @see #setAlpha(double) 
     */
    public ALMA2(double alpha)
    {
        setAlpha(alpha);
    }

    /**
     * Copy constructor
     * @param other the object to copy
     */
    protected ALMA2(ALMA2 other)
    {
        if(other.w != null)
            this.w = other.w.clone();
        this.alpha = other.alpha;
        this.B = other.B;
        this.C = other.C;
        this.k = other.k;
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

    /**
     * Alpha controls the approximation of the large margin formed by ALMA, 
     * with larger values causing more updates. A value of 1.0 will update only
     * on mistakes, while smaller values update if the error was not far enough
     * away from the margin. 
     * <br><br>
     * NOTE: Whenever alpha is set, the value of {@link #setB(double) B} will 
     * also be set to an appropriate value. This is not the only possible value 
     * that will lead to convergence, and can be set manually after alpha is set
     * to another value. 
     * 
     * @param alpha the approximation scale in (0.0, 1.0]
     */
    public void setAlpha(double alpha)
    {
        if(alpha <= 0 || alpha > 1 || Double.isNaN(alpha))
            throw new ArithmeticException("alpha must be in (0, 1], not " + alpha);
        this.alpha = alpha;
        setB(1.0/alpha);
    }

    /**
     * Returns the approximation coefficient used  
     * @return the approximation coefficient used 
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Sets the B variable of the ALMA algorithm, this is set automatically by 
     * {@link #setAlpha(double) }. 
     * @param B the value for B
     */
    public void setB(double B)
    {
        this.B = B;
    }

    /**
     * Returns the B value of the ALMA algorithm
     * @return the B value of the ALMA algorithm
     */
    public double getB()
    {
        return B;
    }

    /**
     * Sets the C value of the ALMA algorithm. The default value is the one 
     * suggested in the paper. 
     * @param C the C value of ALMA
     */
    public void setC(double C)
    {
        if(C <= 0 || Double.isInfinite(C) || Double.isNaN(C))
            throw new ArithmeticException("C must be a posative cosntant");
        this.C = C;
    }

    public double getC()
    {
        return C;
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

    @Override
    public ALMA2 clone()
    {
        return new ALMA2(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes <= 0)
            throw new FailedToFitException("ALMA2 requires numeric features");
        if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("ALMA2 works only for binary classification");
        w = new DenseVector(numericAttributes);
        k = 1;
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final double y_t = targetClass*2-1;
        
        double gamma = B * Math.sqrt(p-1) / k;
        double wx = w.dot(x_t)+bias;
        if(y_t*wx <= (1-alpha)*gamma)//update
        {
            double eta = C/Math.sqrt(p-1)/Math.sqrt(k++);
            w.mutableAdd(eta*y_t, x_t);
            if(useBias)
                bias += eta*y_t;
            final double norm = w.pNorm(2)+bias;
            if(norm > 1)
                w.mutableDivide(norm);
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(w == null)
            throw new UntrainedModelException("The model has not yet been trained");
        double wx = getScore(data);
        CategoricalResults cr =new CategoricalResults(2);
        if(wx < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return w.dot(dp.getNumericalValues());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
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
    
}
