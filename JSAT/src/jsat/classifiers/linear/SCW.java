package jsat.classifiers.linear;

import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import static java.lang.Math.*;
import java.util.List;
import jsat.DataSet;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.Uniform;
import jsat.exceptions.UntrainedModelException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * Provides an Implementation of Confidence-Weighted (CW) learning and Soft 
 * Confidence-Weighted (SCW), both of which are binary linear classifiers 
 * inspired by {@link PassiveAggressive}. The SCW mode handles noisy and 
 * nonlinearly separable datasets better. <br>
 * NOTE: Unlike other online second order methods, when using the full 
 * covariance matrix, all new inputs cost O(d<sup>2</sup>) time to process, even
 * if update is needed. 
 * <br>
 * NOTE: This implementation does not add an implicit bias term, so the solution
 * goes through the origin 
 * <br><br>
 * See:<br>
 * <ul>
 * <li>
 * Crammer, K., Fern, M.,&amp;Pereira, O. (2008). <i>Exact Convex 
 * Confidence-Weighted Learning</i>. In Advances in Neural Information 
 * Processing Systems 22 (pp. 345–352). Retrieved from
 * <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.169.3364">
 * here</a>
 * </li>
 * <li>
 * Wang, J., Zhao, P.,&amp;Hoi, S. C. H. (2012). <i>Exact Soft Confidence-Weighted
 * Learning</i>. ICML. Learning. Retrieved from 
 * <a href="http://arxiv.org/abs/1206.4612">here</a>
 * </li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class SCW extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized, SingleWeightVectorModel
{

    private static final long serialVersionUID = -6721377074407660742L;
    private double C = 1;
    private double eta;
    //all set when eta is set
    private double phi, phiSqrd, zeta, psi;
    private Mode mode;
    private Vec w;
    /**
     * Full covariance matrix
     */
    private Matrix sigmaM;
    /**
     * Diagonal only covariance matrix
     */
    private Vec sigmaV;
    /**
     * Temp vector used to store Sigma * x_t. Make sure the vector is zeroed out
     * before returning from update
     */
    private Vec Sigma_xt;
    
    private boolean diagonalOnly = false;

    /**
     * More than one escape point, makes sure to zero out {@link #Sigma_xt} 
     * using the input incase of sparseness
     * @param x_t 
     */
    private void zeroOutSigmaXt(final Vec x_t)
    {
        //Zero out temp store
       if(diagonalOnly && x_t.isSparse())//only these values will be non zero 
           for(IndexValue iv : x_t)
               Sigma_xt.set(iv.getIndex(), 0.0);
       else
           Sigma_xt.zeroOut();
    }
    
    /**
     * Which version of the algorithms shuld be used
     */
    public static enum Mode
    {
        /**
         * The standard Confidence Weighted algorithm
         */
        CW, 
        /**
         * SCW-I which is strongly related to PA-I
         */
        SCWI,
        /**
         * SCW-II, which is strongly related to PA-II
         */
        SCWII
    }

    /**
     * Creates a new SCW learner
     */
    public SCW()
    {
        this(0.5, Mode.SCWI, true);
    }

    /**
     * Creates a new SCW learner
     * @param eta the margin confidence parameter in [0.5, 1]
     * @param mode mode controlling which algorithm to use
     * @param diagonalOnly whether or not to use only the diagonal of the 
     * covariance matrix
     * @see #setEta(double) 
     * @see #setMode(jsat.classifiers.linear.SCW.Mode) 
     * @see #setDiagonalOnly(boolean) 
     */
    public SCW(double eta, Mode mode, boolean diagonalOnly)
    {
        setEta(eta);
        setMode(mode);
        setDiagonalOnly(diagonalOnly);
    }
    
    /**
     * Copy constructor
     * @param other object to copy
     */
    protected SCW(SCW other)
    {
        this.C = other.C;
        this.diagonalOnly = other.diagonalOnly;
        this.mode = other.mode;
        this.setEta(other.eta);
        if(other.w != null)
            this.w = other.w.clone();
        if(other.sigmaM != null)
            this.sigmaM = other.sigmaM.clone();
        if(other.sigmaV != null)
            this.sigmaV = other.sigmaV.clone();
        if(other.Sigma_xt != null)
            this.Sigma_xt = other.Sigma_xt.clone();
    }

    /**
     * SCW uses a probabilistic version of the margin and attempts to make a 
     * correction so that the confidence with correct label would be of a 
     * certain threshold, which is set by eta. So the threshold must be in 
     * [0.5, 1.0]. Values in the range [0.8, 0.9] often work well on a wide 
     * range of problems
     * 
     * @param eta the confidence to correct to
     */
    public void setEta(double eta)
    {
        if(Double.isNaN(eta) || eta < 0.5 || eta > 1.0)
            throw new IllegalArgumentException("eta must be in [0.5, 1] not " + eta);
        this.eta = eta;
        this.phi = Normal.invcdf(eta, 0, 1);
        this.phiSqrd = phi*phi;
        this.zeta = 1 + phiSqrd;
        this.psi  = 1 + phiSqrd/2;
    }

    /**
     * Returns the target correction confidence
     * @return the target correction confidence
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Set the aggressiveness parameter. Increasing the value of this parameter 
     * increases the aggressiveness of the algorithm. It must be a positive 
     * value. This parameter essentially performs a type of regularization on 
     * the updates
     * <br>
     * The aggressiveness parameter is only used by {@link Mode#SCWI} and 
     * {@link Mode#SCWII}
     * 
     * @param C the positive aggressiveness parameter
     */
    public void setC(double C)
    {
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
     * Controls which version of the algorithm is used
     * @param mode which algorithm to use
     */
    public void setMode(Mode mode)
    {
        this.mode = mode;
    }

    /**
     * Returns which algorithm is used
     * @return which algorithm is used
     */
    public Mode getMode()
    {
        return mode;
    }

    /**
     * Using the full covariance matrix requires <i>O(d<sup>2</sup>)</i> work on 
     * updates, where <i>d</i> is the dimension of the data. Runtime can be 
     * reduced by using only the diagonal of the matrix to perform updates 
     * in <i>O(s)</i> time, where <i>s &le; d</i> is the number of non-zero 
     * values in the input
     * @param diagonalOnly {@code true} to use only the diagonal of the covariance
     */
    public void setDiagonalOnly(boolean diagonalOnly)
    {
        this.diagonalOnly = diagonalOnly;
    }

    /**
     * Returns {@code true} if the covariance matrix is restricted to its diagonal entries
     * @return {@code true} if the covariance matrix is restricted to its diagonal entries
     */
    public boolean isDiagonalOnly()
    {
        return diagonalOnly;
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
    public SCW clone()
    {
        return new SCW(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes <= 0)
            throw new FailedToFitException("SCW requires numeric attributes to perform classification");
        else if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("SCW is a binary classifier");
        w = new DenseVector(numericAttributes);
        Sigma_xt = new DenseVector(numericAttributes);
        if(diagonalOnly)
        {
            sigmaV = new DenseVector(numericAttributes);
            sigmaV.mutableAdd(1);
        }
        else
            sigmaM = Matrix.eye(numericAttributes);
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final double y_t = targetClass*2-1;
        double score = x_t.dot(w);
        
        double v_t = 0;
        if (diagonalOnly)
        {
            //Faster to set only the needed final values
            for (IndexValue iv : x_t)
            {
                double x_t_i = iv.getValue();
                v_t += x_t_i * x_t_i * sigmaV.get(iv.getIndex());
            }

        }
        else
        {
            sigmaM.multiply(x_t, 1, Sigma_xt);
            v_t = x_t.dot(Sigma_xt);
        }
        
        //Check for numerical issues
        
        if(v_t <= 0)//semi positive definit, should not happen
            throw new FailedToFitException("Numerical issues occured");
        
        double m_t = y_t*score;
        
        final double loss = max(0, phi*sqrt(v_t)-m_t);
        
        if(loss <= 1e-15)
        {
            if(!diagonalOnly)
                zeroOutSigmaXt(x_t);
            return;
        }
        final double alpha_t;
        
        
        if(mode == Mode.SCWI || mode == Mode.CW)
        {
            double tmp = max(0, (-m_t*psi+sqrt(m_t*m_t*phiSqrd*phiSqrd/4+v_t*phiSqrd*zeta))/(v_t*zeta) );
            if(mode == Mode.SCWI)
                alpha_t = min(C, tmp);
            else
                alpha_t = tmp;
            
        }
        else//SCWII
        {
            final double n_t = v_t+1/(2*C);
            final double gamma = phi*sqrt(phiSqrd*v_t*v_t*m_t*m_t+4*n_t*v_t*(n_t+v_t*phiSqrd));
            alpha_t = max(0, (-(2*m_t*n_t+phiSqrd*m_t*v_t)+gamma)/(2*(n_t*n_t+n_t*v_t*phiSqrd)));
        }
        
        if(alpha_t < 1e-7)//update is numerically unstable
        {
            if(!diagonalOnly)
                zeroOutSigmaXt(x_t);
            return;
        }
        
        final double u_t = pow(-alpha_t*v_t*phi+sqrt(alpha_t*alpha_t*v_t*v_t*phiSqrd+4*v_t), 2)/4;
            
        
        
        //Now update mean and variance
        if (diagonalOnly)
        {
            for (IndexValue iv : x_t)
            {
                double x_t_i = iv.getValue();
                double tmp = x_t_i * sigmaV.get(iv.getIndex());
                w.increment(iv.getIndex(), alpha_t * y_t * tmp);
            }
        }
        else
            w.mutableAdd(alpha_t * y_t, Sigma_xt);
        
        if(diagonalOnly)//diag does not need beta
        {
            //Only non zeros change the cov values
            final double coef = alpha_t*phi*pow(u_t, -0.5);
            for(IndexValue iv : x_t)
            {
                int idx = iv.getIndex();
                double S_rr = sigmaV.get(idx);
                sigmaV.set(idx, 1/(1/S_rr+coef*pow(iv.getValue(), 2)));
            }
        }
        else
        {
            final double beta_t = alpha_t*phi/(sqrt(u_t)+v_t*alpha_t*phi);
            
            Matrix.OuterProductUpdate(sigmaM, Sigma_xt, Sigma_xt, -beta_t);
            zeroOutSigmaXt(x_t);
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(w == null)
            throw new UntrainedModelException("Model has not yet ben trained");
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
        return w.dot(dp.getNumericalValues());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
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
     * Guess the distribution to use for the regularization term
     * {@link #setC(double) C} .
     *
     * @param d the data set to get the guess for
     * @return the guess for the C parameter
     */
    public static Distribution guessC(DataSet d)
    {
        return new LogUniform(Math.pow(2, -4), Math.pow(2, 4));//from Exact Soft Confidence-Weighted Learning paper
    }
    
    /**
     * Guess the distribution to use for the regularization term
     * {@link #setEta(double) &eta; } .
     *
     * @param d the data set to get the guess for
     * @return the guess for the C parameter
     */
    public static Distribution guessEta(DataSet d)
    {
        return new Uniform(0.5, 0.95);//from Exact Soft Confidence-Weighted Learning paper
    }
}
