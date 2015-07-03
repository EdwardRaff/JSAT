package jsat.classifiers.linear.kernelized;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.classifiers.linear.PassiveAggressive;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;

/**
 * Provides an implementation of Double Update Online Learning (DUOL) algorithm. 
 * It is a kernelized extension of {@link PassiveAggressive} (PA-I) where one 
 * previously learned support vector may be updated upon each addition to the 
 * support vector set. The SV set is unbounded in size. The objective function 
 * is not identical because of the dual updates. 
 * <br><br>
 * Using a larger {@link #setC(double) C} value for DUOL has theoretical 
 * improvements, as it increases the number of "strong" dual updates. The 
 * default value is set to 10 as suggested in the paper. 
 * See:<br>
 * <ul>
 * <li>
 * Zhao, P., Hoi, S. C. H.,&amp;Jin, R. (2011). <i>Double Updating Online Learning</i>.
 * Journal of Machine Learning Research, 12, 1587–1615. Retrieved from 
 * <a href="http://www.cse.msu.edu/~rongjin/publications/zhao11a.pdf"> here</a>
 * </li>
 * <li>
 * Zhao, P., Hoi, S. C. H.,&amp;Jin, R. (2009). <i>DUOL: A Double Updating 
 * Approach for Online Learning</i>. In Y. Bengio, D. Schuurmans, J. Lafferty, 
 * C. K. I. Williams,&amp;A. Culotta (Eds.), 
 * Advances in Neural Information Processing Systems 22 (pp. 2259–2267).
 * </li>
 * </ul>
 * @author Edward Raff
 */
public class DUOL extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized
{
    private static final long serialVersionUID = -4751569462573287056L;
    
    /**
     * Kernel trick to use
     */
    @ParameterHolder
    protected KernelTrick k;
    /**
     * Set of support vectors
     */
    protected List<Vec> S;
    /**
     * Cached outputs of the current decision function on each support vector
     */
    protected List<Double> f_s;
    
    /**
     * Signed weights for each support vector. Original paper uses the notation 
     * of gamma in the original paper, but this is weird and easily confused 
     * with the sign values y_i. The sign class values can be obtained from the 
     * signed alphas using {@link Math#signum(double) }
     */
    protected List<Double> alphas;
    protected List<Double> accelCache;
    /**
     * Stores the values of k(x_i, y) for reuse when observing a new example
     */
    protected DoubleList kTmp;
    
    protected double rho = 0;
    protected double C = 10;

    /**
     * Creates a new DUOL learner
     * @param k the kernel to use
     */
    public DUOL(KernelTrick k)
    {
        this.k = k;
        this.S = new ArrayList<Vec>();
        this.f_s = new DoubleList();
        this.alphas = new DoubleList();
    }

    /**
     * Copy constructor
     * @param other the object to copy
     */
    protected DUOL(DUOL other)
    {
        this.k = other.k.clone();
        if(other.S != null)
        {
            this.S = new ArrayList<Vec>(other.S.size());
            for(Vec v : other.S)
                this.S.add(v.clone());
            this.f_s = new DoubleList(other.f_s);
            this.alphas = new DoubleList(other.alphas);
            if(other.accelCache != null)
                this.accelCache = new DoubleList(other.accelCache);
            if(other.kTmp != null)
                this.kTmp = new DoubleList(other.kTmp);
        }
        this.rho = other.rho;
        this.C = other.C;
    }

    @Override
    public DUOL clone()
    {
        return new DUOL(this);
    }

    /**
     * Sets the aggressiveness parameter. Increasing the value of this parameter 
     * increases the aggressiveness of the algorithm. It must be a positive 
     * value. This parameter essentially performs a type of regularization on 
     * the updates
     * @param C the aggressiveness parameter in (0, Inf)
     */
    public void setC(double C)
    {
        if(Double.isNaN(C) || C <= 0 || Double.isInfinite(C))
            throw new IllegalArgumentException("C parameter must be in range (0, inf) not " + C);
        this.C = C;
    }

    /**
     * Returns the aggressiveness parameter
     * @return the aggressiveness parameter
     */
    public double getC()
    {
        return C;
    }

    /**
     * Sets the "conflict" parameter, which controls how often double updates 
     * are performed. Smaller (near zero) values tend to produce more double 
     * updates, with values near 1 producing few double updates. The value must 
     * be in the range [0, 1]
     * @param rho the conflict parameter for when to update a second support vector
     */
    public void setRho(double rho)
    {
        this.rho = rho;
    }

    /**
     * Returns the "conflict" parameter value for the threshold of performing double updates
     * @return the "conflict" parameter value for the threshold of performing double updates
     */
    public double getRho()
    {
        return rho;
    }

    /**
     * Sets the kernel trick to use
     * @param k the kernel trick to use
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

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes <= 0)
            throw new FailedToFitException("DUOL requires numeric features");
        else if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("DUOL supports only binnary classification");
        
        this.S = new ArrayList<Vec>();
        this.f_s = new DoubleList();
        this.alphas = new DoubleList();
        this.accelCache = new DoubleList();
        this.kTmp = new DoubleList();
    }

    @Override
    public synchronized void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final double y_t = targetClass*2-1;
        final List<Double> qi = k.getQueryInfo(x_t);
        double score = score(x_t, qi, true);

        final double loss_t = max(0, 1-y_t*score);
        
        if(loss_t <= 0)
            return;

        //start of line 8:
        int b = -1;
        double w_min = Double.POSITIVE_INFINITY;
        for(int i = 0; i < S.size(); i++)
        {
            if(f_s.get(i) <= 1)
            {
                double tmp = signum(alphas.get(i))*y_t*kTmp.get(i);
                if(tmp <= w_min)
                {
                    w_min = tmp;
                    b = i;
                }
            }
        }
        
        final double k_t = k.eval(0, 0, Arrays.asList(x_t), qi);
        if(w_min <= -rho)
        {
            
            final double k_b = k.eval(b, b, S, accelCache);
            final double k_tb = kTmp.get(b);
            final double alpha_b = alphas.get(b);
            final double w_tb = y_t*signum(alpha_b)*k_tb;
            final double gamma_hat_b = abs(alpha_b);
            final double loss_b = 1-signum(alpha_b)*f_s.get(b);
            
            //(C-gamma_hat_b) is common expression bellow, so use this insted
            final double CmGhb = (C-gamma_hat_b);

            final double gamma_t;
            final double gamma_b;
            final double gamma_b_delta;
            
            if(k_t*C +w_tb*CmGhb-loss_t < 0 && k_b*CmGhb+w_tb*C-loss_b < 0)
            {
                gamma_t = C;
                gamma_b_delta = CmGhb;
            }
            else if(  (w_tb*w_tb*C-w_tb*loss_b-k_t*k_b*C+k_b*loss_t)/k_b > 0 && isIn((loss_b-w_tb*C)/k_b, -gamma_hat_b, CmGhb) )
            {
                gamma_t = C;
                gamma_b_delta = (loss_b-w_tb*C)/k_b;
            }
            else if(  isIn((loss_t-w_tb*CmGhb)/k_t, 0, C) && loss_b-k_b*CmGhb-w_tb*(loss_t-w_tb*CmGhb)/k_t > 0  )
            {
                gamma_t = (loss_t-w_tb*CmGhb/k_t);
                gamma_b_delta = CmGhb;
            }
            else//last case is the only option by elimination, no need to write complicated if statment for it
            {
                final double denom = k_t*k_b-w_tb*w_tb;
                gamma_t       = (k_b*loss_t-w_tb*loss_b)/denom;
                gamma_b_delta = (k_t*loss_b-w_tb*loss_t)/denom;
            }

            gamma_b = gamma_hat_b+gamma_b_delta;
            
            //add new SV
            S.add(x_t);
            accelCache.addAll(qi);
            kTmp.add(k_t);
            alphas.add(y_t*gamma_t);
            //dont forget curretn SV self value which gets updated in the loop
            f_s.add(score);
            
            for(int i = 0; i < S.size(); i++)
            {
                final double y_i = signum(alphas.get(i));
                f_s.set(i, f_s.get(i)+y_i*gamma_t*y_t*kTmp.get(i)+y_i*gamma_b_delta*signum(alpha_b)*k.eval(i, b, S, accelCache));
            }
            
            //update old weight for b
            alphas.set(b, signum(alpha_b)*gamma_b);
        }
        else  /* no auxiliary example found */
        {
            final double gamma_t = min(C, loss_t/k_t);
            
            //add new SV
            S.add(x_t);
            accelCache.addAll(qi);
            kTmp.add(k_t);
            alphas.add(y_t*gamma_t);
            //dont forget curretn SV self value which gets updated in the loop
            f_s.add(score);
            
            for(int i = 0; i < S.size(); i++)
            {
                final double y_i = signum(alphas.get(i));
                f_s.set(i, f_s.get(i)+y_i*gamma_t*y_t*kTmp.get(i));
            }
        }   
    }
    
    private boolean isIn(double x, double a, double b)
    {
        return a <= x && x <= b;
    }
    
    private double score(Vec x, List<Double> qi, boolean store)
    {
        if(store)
            kTmp.clear();
        double score = 0;
        for(int i = 0; i < S.size(); i++)
        {
            double tmp = k.eval(i, x, qi, S, accelCache);
            if(store)
                kTmp.add(tmp);
            score += alphas.get(i)*tmp;
        }
        return score;
    }
    
    private double score(Vec x, List<Double> qi)
    {
        return score(x, qi, false);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(alphas == null)
            throw new UntrainedModelException("Model has not yet been trained");
        CategoricalResults cr = new CategoricalResults(2);
        double score = getScore(data);
        if(score < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        Vec x = dp.getNumericalValues();
        return score(x, k.getQueryInfo(x));
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
    /**
     * Guesses the distribution to use for the C parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the C parameter
     * @see #setC(double)
     */
    public static Distribution guessC(DataSet d)
    {
        return new LogUniform(1e-4, 1e5);
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
