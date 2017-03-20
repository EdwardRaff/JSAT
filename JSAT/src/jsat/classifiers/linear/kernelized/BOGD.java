package jsat.classifiers.linear.kernelized;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.Vec;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.LossC;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * Bounded Online Gradient Descent (BOGD) is a kernel learning algorithm that 
 * uses a bounded number of support vectors. Once the maximum number of support
 * vectors is reached, old vectors are dropped either in a uniform random 
 * fashion, or weighted by the kernel function and the current coefficient for 
 * the vector. The later is the default method and is referred to as BOGD++.<br>
 * <br>
 * See: Zhao, P., Wang, J., Wu, P., Jin, R.,&amp;Hoi, S. C. H. (2012). <i>Fast
 * Bounded Online Gradient Descent Algorithms for Scalable Kernel-Based Online
 * Learning</i>. In Proceedings of the 29th International Conference on Machine 
 * Learning (pp. 169–176). Learning; Machine Learning. Retrieved from 
 * <a href="http://arxiv.org/abs/1206.4633">here</a>
 * @author Edward Raff
 */
public class BOGD extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized
{

    private static final long serialVersionUID = -3547832514098781996L;
    @ParameterHolder
    private KernelTrick k;
    private int budget;
    private double eta;
    private double reg;
    private double maxCoeff;
    private LossC lossC;
    
    private boolean uniformSampling;
    
    private Random rand;
    private List<Vec> vecs;
    /**
     * Stores the sqrt of each support vector's kernel product with itself
     */
    private List<Double> selfK;
    private DoubleList alphas;
    private List<Double> accelCache;
    /**
     * Cache of values used for BOGD++ sampling
     */
    private double[] dist;
    
    /**
     * Creates a new BOGD++ learner using the {@link HingeLoss}
     * @param k the kernel trick to use
     * @param budget the budget for support vectors to allow
     * @param eta the learning rate to use
     * @param reg the regularization parameter
     * @param maxCoeff the maximum support vector coefficient to allow
     */
    public BOGD(KernelTrick k, int budget, double eta, double reg, double maxCoeff)
    {
        this(k, budget, eta, reg, maxCoeff, new HingeLoss());
    }

    /**
     * Creates a new BOGD++ learner
     * @param k the kernel trick to use
     * @param budget the budget for support vectors to allow
     * @param eta the learning rate to use
     * @param reg the regularization parameter
     * @param maxCoeff the maximum support vector coefficient to allow
     * @param lossC the loss function to use
     */
    public BOGD(KernelTrick k, int budget, double eta, double reg, double maxCoeff, LossC lossC)
    {
        setKernel(k);
        setBudget(budget);
        setEta(eta);
        setRegularization(reg);
        setMaxCoeff(maxCoeff);
        this.lossC = lossC;
        setUniformSampling(false);
    }

    /**
     * Copy constructor
     * @param toCopy the object to make a copy of
     */
    public BOGD(BOGD toCopy)
    {
        this.k = toCopy.k.clone();
        this.budget = toCopy.budget;
        this.eta = toCopy.eta;
        this.reg = toCopy.reg;
        this.maxCoeff = toCopy.maxCoeff;
        this.lossC = toCopy.lossC.clone();
        this.uniformSampling = toCopy.uniformSampling;
        this.rand = RandomUtil.getRandom();
        if(toCopy.vecs != null)
        {
            this.vecs = new ArrayList<Vec>(budget);
            for(Vec v : toCopy.vecs)
                this.vecs.add(v.clone());
            this.selfK = new DoubleList(toCopy.selfK);
            this.alphas = new DoubleList(toCopy.alphas);
        }
        if(toCopy.accelCache != null)
            this.accelCache = new DoubleList(toCopy.accelCache);
        if(toCopy.dist != null)
            this.dist = Arrays.copyOf(toCopy.dist, toCopy.dist.length);
    }
    
    /**
     * Sets the regularization parameter used for training. The original paper
     * suggests values in the range 2<sup>x</sup>/T<sup>2</sup> for <i>x</i> 
     * &isin; {-3, -2, -1, 0, 1, 2, 3} where <i>T</i> is the number of data 
     * instances that will be trained on
     * 
     * @param regularization the positive regularization parameter to use. 
     */
    public void setRegularization(double regularization)
    {
        if(regularization <= 0 || Double.isNaN(regularization) || Double.isInfinite(regularization))
            throw new IllegalArgumentException("Regularization must be positive, not " + regularization);
        this.reg = regularization;
    }

    /**
     * Returns the regularization parameter used
     * @return the regularization parameter used
     */
    public double getRegularization()
    {
        return reg;
    }

    /**
     * Sets the learning rate to use for training. The original paper suggests 
     * values in the range 2<sup>x</sup> for <i>x</i> 
     * &isin; {-3, -2, -1, 0, 1, 2, 3}
     * @param eta the positive learning rate to use
     */
    public void setEta(double eta)
    {
        if(eta <= 0 || Double.isNaN(eta) || Double.isInfinite(eta))
            throw new IllegalArgumentException("Eta must be positive, not " + eta);
        this.eta = eta;
    }

    /**
     * Returns the learning rate in use
     * @return the learning rate in use
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Sets the maximum allowed value for any support vector allowed. The 
     * original paper suggests values in the range 2<sup>x</sup> for <i>x</i> 
     * &isin; {0, 1, 2, 3, 4}
     * @param maxCoeff the maximum value for any support vector
     */
    public void setMaxCoeff(double maxCoeff)
    {
        if(maxCoeff <= 0 || Double.isNaN(maxCoeff) || Double.isInfinite(maxCoeff))
            throw new IllegalArgumentException("MaxCoeff must be positive, not " + maxCoeff);
        this.maxCoeff = maxCoeff;
    }

    /**
     * Returns the maximum allowed value for any support vector 
     * @return the maximum allowed value for any support vector 
     */
    public double getMaxCoeff()
    {
        return maxCoeff;
    }

    /**
     * Sets the budget for support vectors
     * @param budget the allowed budget for support vectors
     */
    public void setBudget(int budget)
    {
        if(budget <= 0 )
            throw new IllegalArgumentException("Budget must be positive, not " + budget);
        this.budget = budget;
    }

    /**
     * Returns the maximum number of allowed support vectors
     * @return the maximum number of allowed support vectors
     */
    public int getBudget()
    {
        return budget;
    }

    /**
     * Sets the kernel to use
     * @param k the kernel to use
     */
    public void setKernel(KernelTrick k)
    {
        this.k = k;
    }

    /**
     * Returns the kernel to use
     * @return the kernel to use
     */
    public KernelTrick getKernel()
    {
        return k;
    }

    /**
     * Sets whether or not support vectors should be removed by uniform sampling
     * or not. The default is {@code false}, which corresponds to BOGD++. 
     * @param uniformSampling {@code true} to use uniform sampling, 
     * {@code false} otherwise. 
     */
    public void setUniformSampling(boolean uniformSampling)
    {
        this.uniformSampling = uniformSampling;
    }

    /**
     * Returns {@code true } is uniform sampling is in use, or {@code false} if 
     * the BOGD++ sampling procedure is in use
     * @return {@code true } is uniform sampling is in use, or {@code false} if 
     * the BOGD++ sampling procedure is in use
     */
    public boolean isUniformSampling()
    {
        return uniformSampling;
    }
    
    @Override
    public BOGD clone()
    {
        return new BOGD(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        vecs = new ArrayList<Vec>(budget);
        alphas = new DoubleList(budget);
        selfK = new DoubleList(budget);
        if(k.supportsAcceleration())
            accelCache = new DoubleList(budget);
        else
            accelCache = null;
        if(!uniformSampling)
            dist = new double[budget];
        rand = RandomUtil.getRandom();
    }
    
    @Override
    public double getScore(DataPoint dp)
    {
        Vec x = dp.getNumericalValues();
        return score(x, k.getQueryInfo(x));
    }
    
    private double score(Vec x, List<Double> qi)
    {
        return k.evalSum(vecs, accelCache, alphas.getBackingArray(), x, qi, 0, alphas.size());
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final double y_t = targetClass*2-1;
        
        final List<Double> qi = k.getQueryInfo(x_t);
        final double score = score(x_t, qi);
        final double lossD = lossC.getDeriv(score, y_t);

        if(lossD == 0)
        {
            alphas.getVecView().mutableMultiply(1-eta*reg);
        }
        else
        {
            if(vecs.size() < budget)
            {
                alphas.getVecView().mutableMultiply(1-eta*reg);
                alphas.add(-eta*lossD);
                selfK.add(Math.sqrt(k.eval(0, 0, Arrays.asList(x_t), qi)));
                if(k.supportsAcceleration())
                    accelCache.addAll(qi);
                vecs.add(x_t);
            }
            else//budget maintinance
            {
                final int toRemove;
                final double normalize;
                if(uniformSampling)
                {
                    toRemove = rand.nextInt(budget);
                    normalize = 1;
                }
                else
                {
                    double s = 0;
                    for(int i = 0; i < budget; i++)
                        s += Math.abs(alphas.get(i))*selfK.get(i);
                    s = (budget-1)/s;
                    final double target = rand.nextDouble();
                    double cur = 0;
                    int i = -1;
                    while(cur < target)
                    {
                        i++;
                        cur += dist[i] = 1-s*alphas.get(i)*selfK.get(i);
                    }
                    toRemove = i++;
                    while(i < budget)
                        cur += dist[i] = 1-s*alphas.get(i)*selfK.get(i++);
                    normalize = cur;
                }
                
                for(int i = 0; i < budget; i++)
                {
                    if(i == toRemove)
                        continue;
                    double alpha_i = alphas.getD(i);
                    double sign = Math.signum(alpha_i);
                    alpha_i = Math.abs(alpha_i);
                    double tmp = uniformSampling ? 1.0/budget : dist[i]/normalize;
                    alphas.set(i, sign*Math.min((1-reg*eta)/(1-tmp)*alpha_i, maxCoeff*eta));
                }
                
                //Remove old point
                if(k.supportsAcceleration())
                {
                    int catToRet = accelCache.size()/budget;
                    for(int i = 0; i < catToRet; i++)
                        accelCache.remove(toRemove*catToRet);
                }
                alphas.remove(toRemove);
                vecs.remove(toRemove);
                selfK.remove(toRemove);
                
                //Add new point
                alphas.add(-eta*lossD);
                selfK.add(Math.sqrt(k.eval(0, 0, Arrays.asList(x_t), qi)));
                accelCache.addAll(qi);
                vecs.add(x_t);
            }
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        return lossC.getClassification(score(x, k.getQueryInfo(x)));
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
    /**
     * Guesses the distribution to use for the Regularization parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the Regularization parameter
     * @see #setRegularization(double) 
     */
    public static Distribution guessRegularization(DataSet d)
    {
        double T2 = d.getSampleSize();
        T2*=T2;
        
        return new LogUniform(Math.pow(2, -3)/T2, Math.pow(2, 3)/T2);
    }
    
    /**
     * Guesses the distribution to use for the &eta; parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the &eta; parameter
     * @see #setEta(double) 
     */
    public static Distribution guessEta(DataSet d)
    {
        return new LogUniform(Math.pow(2, -3), Math.pow(2, 3));
    }
    
    /**
     * Guesses the distribution to use for the MaxCoeff parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the MaxCoeff parameter
     * @see #setMaxCoeff(double) (double) 
     */
    public static Distribution guessMaxCoeff(DataSet d)
    {
        return new LogUniform(Math.pow(2, 0), Math.pow(2, 4));
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
