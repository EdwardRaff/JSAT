package jsat.classifiers.linear.kernelized;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.random.XORWOW;
import static java.lang.Math.*;
import jsat.DataSet;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.utils.random.RandomUtil;

/**
 * An implementation of Conservative Stochastic Kernel Logistic Regression. This
 * is an online algorithm that obtains sparse solutions by conservatively 
 * rejecting updates based on a binomial distribution of the error on each 
 * update. <br><br>
 * This algorithm works best on data sets with a very high number of samples 
 * where a high accuracy is obtainable using a kernel model. It is often the 
 * case that this model produces accurate results, but has a low confidence due 
 * to the conservative updating. This can be counteracted by having a very large
 * number of features, but that often increases the size of the model. 
 * <br><br>
 * It is important to read the documentation and test some different values for
 * the {@link #setEta(double) learning rate} and {@link #setGamma(double) gamma} 
 * variables. They behave different compared to many algorithms. 
 * <br><br>
 * It is possible to obtain a more confident model and a slightly larger model 
 * by using several epochs. Instead of using this class, the 
 * {@link CSKLRBatch batch version} of this algorithm should be used instead. 
 * <br><br>
 * See paper: <br>
 * Zhang, L., Jin, R., Chen, C., Bu, J.,&amp;He, X. (2012). <i>Efficient Online 
 * Learning for Large-Scale Sparse Kernel Logistic Regression</i>. Twenty-Sixth 
 * AAAI Conference on Artificial Intelligence (pp. 1219–1225). Retrieved from 
 * <a href="http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPDFInterstitial/5003/5544">here</a>
 * 
 * @author Edward Raff
 */
public class CSKLR extends BaseUpdateableClassifier implements Parameterized
{

	private static final long serialVersionUID = 2325605193408720811L;
	private double eta;
    private DoubleList alpha;
    private List<Vec> vecs;
    private double curNorm;
    private KernelTrick k;
    private double R;
    private Random rand;
    private UpdateMode mode;
    private double gamma = 2;
    private List<Double> accelCache;

    /**
     * Creates a new CSKLR object
     * @param eta the learning rate to use
     * @param k the kernel trick to use
     * @param R the maximal norm of the surface 
     * @param mode the mode to use
     */
    public CSKLR(double eta, KernelTrick k, double R, UpdateMode mode)
    {
        setEta(eta);
        setKernel(k);
        setR(R);
        setMode(mode);
    }
   
    /**
     * Guesses the distribution to use for the R parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the R parameter
     * @see #setR(double) 
     */
    public static Distribution guessR(DataSet d)
    {
        return new LogUniform(1, 1e5);
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
    
    /**
     * Controls when updates are performed on the model. Depending on which 
     * update model is used, the acceptable values and behaviors of 
     * {@link #setEta(double) } and {@link #setGamma(double) } may change. 
     * <br><br>
     * The "auxiliary" modes perform updates with a probability of 
     * <i>log(1+e<sup>-z</sup>) / a(z)</i>, where <i>a(z)</i> is the auxiliary 
     * function and <i>z</i> is the raw margin value times the class label. 
     */
    public enum UpdateMode 
    {

        /**
         * NC stands for Non-Conservative, this mode will perform a model update
         * on every new input, creating a very dense model. The 
         * {@link #setEta(double) learning rate} may take on any positive value,
         * and {@link #setGamma(double) } is not used. 
         */
        NC
        {
            @Override
            protected double pt(double y, double score, double preScore, double eta, double gamma)
            {
                return 1;
            }

            @Override
            protected double grad(double y, double score, double preScore, double gamma)
            {
                return score-1;
            }
        },
        /**
         * Performs model updates probabilistically based on their distance from
         * the margin of the classifier. In this case, {@link #setEta(double) } 
         * should be less than 2, or the model will become dense. 
         * {@link #setGamma(double) } is not used. 
         */
        MARGIN
        {
            @Override
            protected double pt(double y, double score, double preScore, double eta, double gamma)
            {
                return (2-eta)/(2-eta+eta*score);
            }

            @Override
            protected double grad(double y, double score, double preScore, double gamma)
            {
                return score-1;
            }
        },
        /**
         * Performs model updates based on a "auxiliary" function
         * <i>a(z) = log(&gamma; + e<sup>-z</sup>)</i>. 
         * {@link #setGamma(double) gamma} should be in the range (1, Infinity)
         * where larger values increase the sparsity of the model
         * <br><br>
         * This is the main auxiliary method used by the authors. They use 
         * values for &gamma; in the range of <i>1+10<sup>&plusmn; 
         * x</sup></i> &forall; x &isin; {0, 1, 2, 3, 4}
         */
        AUXILIARY_1
        {
            @Override
            protected double pt(double y, double score, double preScore, double eta, double gamma)
            {
                double z = y*preScore;
                return log(1+exp(-z))/log(gamma+exp(-z));
            }

            @Override
            protected double grad(double y, double score, double preScore, double gamma)
            {
                double z = y*preScore;
                return -1/(1+gamma*exp(z));
            }
        },
        /**
         * Performs model updates based on a "auxiliary" function
         * <i>a(z) = log(1 + &gamma;  e<sup>-z</sup>)</i>. 
         * {@link #setGamma(double) gamma} should be in the range (1, Infinity)
         * where larger values increase the sparsity of the model
         */
        AUXILIARY_2
        {
            @Override
            protected double pt(double y, double score, double preScore, double eta, double gamma)
            {
                double z = y*preScore;
                return log(1+exp(-z))/log(1+gamma*exp(-z));
            }

            @Override
            protected double grad(double y, double score, double preScore, double gamma)
            {
                double z = y*preScore;
                return -gamma/(gamma+exp(z));
            }
        },
        /**
         * Performs model updates based on a "auxiliary" function
         * <i>a(z) = max(loss(z), loss(&gamma;)</i>. 
         * {@link #setGamma(double) gamma} should be in the range (0, Infinity)
         * where smaller values increase the sparsity of the model
         */
        AUXILIARY_3
        {
            @Override
            protected double pt(double y, double score, double preScore, double eta, double gamma)
            {
                double z = y*preScore;
                return log(1+exp(-z))/log(1+exp(-gamma));
            }

            @Override
            protected double grad(double y, double score, double preScore, double gamma)
            {
                return score-1;
            }
        };

        /**
         * Returns the Bernoulli trial probability variable 
         * @param y the sign of the input point
         * @param score the logistic regression score for the input
         * @param preScore the raw margin before the final
         * @param eta the learning rate
         * @param gamma the gamma variable
         * @return the Bernoulli trial probability variable  
         */
        abstract protected double pt(double y, double score, double preScore, double eta, double gamma);
        
        /**
         * Get the gradient value that should be applied based on the input 
         * variable from the current model 
         * @param y the sign of the input point
         * @param score the logistic regression score for the input
         * @param preScore the raw margin before the final
         * @param gamma the gamma variable
         * @return the coefficient to apply to the stochastic update
         */
        abstract protected double grad(double y, double score, double preScore, double gamma);
    }

    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    protected CSKLR(CSKLR toClone)
    {
        if(toClone.alpha != null)
            this.alpha = new DoubleList(toClone.alpha);
        if(toClone.vecs != null)
        {
            this.vecs = new ArrayList<Vec>(toClone.vecs);
        }
        this.curNorm = toClone.curNorm;
        this.mode = toClone.mode;
        this.R = toClone.R;
        this.eta = toClone.eta;
        this.setKernel(toClone.k.clone());
        if(toClone.accelCache != null)
            this.accelCache = new DoubleList(toClone.accelCache);
        this.gamma = toClone.gamma;
        this.rand = RandomUtil.getRandom();
        this.setEpochs(toClone.getEpochs());
    }

    /**
     * Sets the learning rate to use for the algorithm. Unlike many other 
     * stochastic algorithms, the learning rate for CSKLR should be large, often
     * in the range of (0.5, 1) - and can even be larger than 1 at times. If the
     * learning rate is too low, it may be difficult to get strong confidence 
     * results from the algorithm. 
     * 
     * @param eta the positive learning rate to use
     */
    public void setEta(double eta)
    {
        if(eta < 0 || Double.isNaN(eta) || Double.isInfinite(eta))
            throw new IllegalArgumentException("The learning rate should be in (0, Inf), not " + eta);
        this.eta = eta;
    }

    /**
     * Returns the learning rate to use
     * @return the learning rate to use
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Sets the maximal margin norm value for the algorithm. When the norm is 
     * exceeded, the coefficients will be rescaled to fit in the norm. If the 
     * maximal norm is too small (less than 5), it may be difficult to get 
     * strong confidence results from the algorithm. <br>
     * A good range of values suggested by the original paper is 10<sup>x</sup>
     * &forall; x &isin; {0, 1, 2, 3, 4, 5}
     * @param R 
     */
    public void setR(double R)
    {
        if(R < 0 || Double.isNaN(R) || Double.isInfinite(R))
            throw new IllegalArgumentException("The max norm should be in (0, Inf), not " + R);
        this.R = R;
    }

    /**
     * Returns the maximal norm of the algorithm
     * @return the maximal norm of the algorithm
     */
    public double getR()
    {
        return R;
    }

    /**
     * Sets what update mode should be used. The update mode controls the 
     * sparsity of the mode, and the behavior of {@link #setGamma(double) }
     * @param mode the update mode to use
     */
    public void setMode(UpdateMode mode)
    {
        this.mode = mode;
    }

    /**
     * Returns the update mode in use
     * @return  the update mode in use
     */
    public UpdateMode getMode()
    {
        return mode;
    }

    /**
     * Sets the gamma value to use. This value, depending on which 
     * {@link UpdateMode} is used, controls the sparsity of the model.
     * @param gamma the gamma parameter, which is at least always positive
     */
    public void setGamma(double gamma)
    {
        if(gamma < 0 || Double.isNaN(gamma) || Double.isInfinite(gamma))
            throw new IllegalArgumentException("Gamma must be in (0, Infity), not " + gamma);
        this.gamma = gamma;
    }

    /**
     * Returns the gamma sparsity parameter value
     * @return the gamma sparsity parameter value
     */
    public double getGamma()
    {
        return gamma;
    }

    /**
     * Set which kernel trick to use 
     * @param k the kernel to use
     */
    public void setKernel(KernelTrick k)
    {
        this.k = k;
    }

    /**
     * Returns the kernel trick in use
     * @return the kernel trick in use
     */
    public KernelTrick getKernel()
    {
        return k;
    }
    
    /**
     * Computes the margin score for the given data point
     * @param x the input vector
     * @return the margin score
     */
    private double getPreScore(Vec x)
    {
        return k.evalSum(vecs, accelCache, alpha.getBackingArray(), x, 0, alpha.size());
    }
    
    /**
     * Returns the binary logistic regression score
     * @param y the sign of the desired class (-1 or 1)
     * @param pre the raw coefficient score
     * @return the probability in [0, 1] that the score is of the desired class
     */
    protected static double getScore(double y, double pre)
    {
        return 1/(1+Math.exp(-y*pre));
    }

    @Override
    public CSKLR clone()
    {
        return new CSKLR(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("CSKLR supports only binary classification");
        alpha = new DoubleList();
        vecs = new ArrayList<Vec>();
        curNorm = 0;
        rand = RandomUtil.getRandom();
        if(k.supportsAcceleration())
            accelCache = new DoubleList();
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        double y_t = targetClass*2-1;
        Vec x_t = dataPoint.getNumericalValues();
        double pre = getPreScore(x_t);
        double score = getScore(y_t, pre);
        
        switch(mode)
        {
            case NC:
                break;
            default:
                double pt = mode.pt(y_t, score, pre, eta, gamma);
                if(rand.nextDouble() > pt)
                    return;
             break;   
        }
        
        
        double alpha_i = -eta*y_t*mode.grad(y_t, score, pre, gamma)*dataPoint.getWeight();

        alpha.add(alpha_i);
        vecs.add(x_t);
        k.addToCache(x_t, accelCache);
        curNorm += Math.abs(alpha_i) * k.eval(vecs.size(), vecs.size(), vecs, accelCache);

        //projection step
        if (curNorm > R)
        {
            double coef = R/curNorm;
            for(int i = 0; i < alpha.size(); i++)
                alpha.set(i, alpha.get(i)*coef);
            curNorm = coef;
        }
        
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        
        double p_0 = getScore(-1, getPreScore(data.getNumericalValues()));

        cr.setProb(0, p_0);
        cr.setProb(1, 1-p_0);
        
        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
}
