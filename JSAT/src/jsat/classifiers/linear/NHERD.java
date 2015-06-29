package jsat.classifiers.linear;

import java.util.List;
import jsat.DataSet;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * Implementation of the Normal Herd (NHERD) algorithm for learning a linear 
 * binary classifier. It is related to both {@link AROW} and the PA-II variant 
 * of {@link PassiveAggressive}. <br>
 * Unlike similar algorithms, several methods of using only the diagonal values
 * of the covariance are available. 
 * <br>
 * NOTE: This implementation does not add an implicit bias term, so the solution
 * goes through the origin 
 * <br><br>
 * See:<br>
 * Crammer, K.,&amp;Lee, D. D. (2010). <i>Learning via Gaussian Herding</i>.
 * Pre-proceeding of NIPS (pp. 451–459). Retrieved from 
 * <a href="http://webee.technion.ac.il/Sites/People/koby/publications/gaussian_mob_nips10.pdf">
 * here</a>
 * 
 * @author Edward Raff
 */
public class NHERD extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized, SingleWeightVectorModel
{

    private static final long serialVersionUID = -1186002893766449917L;
    private Vec w;
    /**
     * Full covariance matrix
     */
    private Matrix sigmaM;
    /**
     * Diagonal only covariance matrix
     */
    private Vec sigmaV;
    
    private CovMode covMode;
    private double C;
    
    /**
     * Temp vector used to store Sigma * x_t
     */
    private Vec Sigma_xt;
    
    /**
     * Sets what form of covariance matrix to use
     */
    public static enum CovMode
    {
        /**
         * Use the full covariance matrix
         */
        FULL, 
        /**
         * Standard diagonal method, only the diagonal values get updated by 
         * dropping the other terms. 
         */
        DROP, 
        /**
         * Creates the diagonal by dropping the terms of the inverse of the 
         * covariance matrix that is used to perform the update. This authors 
         * suggest this is usually the best for diagonal covariance matrices 
         * from empirical testing. 
         */
        PROJECT, 
        /**
         * Creates the diagonal by solving the derivative with respect to the 
         * specific objective function of NHERD
         */
        EXACT
    }
    
    /**
     * Creates a new NHERD learner
     * @param C the aggressiveness parameter 
     * @param covMode how to form the covariance matrix
     * @see #setC(double) 
     * @see #setCovMode(jsat.classifiers.linear.NHERD.CovMode) 
     */
    public NHERD(double C, CovMode covMode)
    {
        setC(C);
        setCovMode(covMode);
    }
    
    /**
     * Copy constructor
     * @param other the object to copy
     */
    protected NHERD(NHERD other)
    {
        this.C = other.C;
        this.covMode = other.covMode;
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
     * Set the aggressiveness parameter. Increasing the value of this parameter 
     * increases the aggressiveness of the algorithm. It must be a positive 
     * value. This parameter essentially performs a type of regularization on 
     * the updates
     * 
     * @param C the positive aggressiveness parameter
     */
    public void setC(double C)
    {
        if(Double.isNaN(C) || Double.isInfinite(C) || C <= 0)
            throw new IllegalArgumentException("C must be a postive constant, not " + C);
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
     * Sets the way in which the covariance matrix is formed. If using the full 
     * covariance matrix, rank-1 updates mean updates to the model take 
     * <i>O(d<sup>2</sup>)</i> time, where <i>d</i> is the dimension of the 
     * input. Runtime can be reduced by using only the diagonal of the matrix to
     * perform updates in <i>O(s)</i> time, where <i>s &le; d</i> is the number 
     * of non-zero values in the input
     * 
     * @param covMode the way to form the covariance matrix
     */
    public void setCovMode(CovMode covMode)
    {
        this.covMode = covMode;
    }

    /**
     * Returns the mode for forming the covariance 
     * @return the mode for forming the covariance 
     */
    public CovMode getCovMode()
    {
        return covMode;
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
    public NHERD clone()
    {
        return new NHERD(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes <= 0)
            throw new FailedToFitException("AROW requires numeric attributes to perform classification");
        else if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("AROW is a binary classifier");
        w = new DenseVector(numericAttributes);
        Sigma_xt = new DenseVector(numericAttributes);
        if(covMode != CovMode.FULL)
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
        Vec x_t = dataPoint.getNumericalValues();
        double y_t = targetClass*2-1;
        double pred = x_t.dot(w);
        if(y_t*pred > 1)
            return;//No update needed
        //else, wrong label or margin too small
        
        double alpha;
        if(covMode != CovMode.FULL)
        {
            alpha = 0;
            //Faster to set only the needed final values
            for (IndexValue iv : x_t)
            {
                double x_ti = iv.getValue();
                alpha += x_ti * x_ti * sigmaV.get(iv.getIndex());
            }
        }
        else
        {
            sigmaM.multiply(x_t, 1, Sigma_xt);
            alpha = x_t.dot(Sigma_xt);
        }
        
        final double loss = Math.max(0, 1 - y_t * pred);
        final double w_c = y_t * loss / (alpha + 1 / C);
        
        if (covMode == CovMode.FULL)
            w.mutableAdd(w_c, Sigma_xt);
        else
            for (IndexValue iv : x_t)
                w.increment(iv.getIndex(), w_c * iv.getValue() * sigmaV.get(iv.getIndex()));
        
        double numer = C*(C*alpha+2);
        double denom = (1+C*alpha)*(1+C*alpha);
        switch (covMode)
        {
            
            case FULL:
                Matrix.OuterProductUpdate(sigmaM, Sigma_xt, Sigma_xt, -numer/denom);
                break;
            case DROP:
                final double c = -numer/denom;
                for (IndexValue iv : x_t)
                {
                    int idx = iv.getIndex();
                    double x_ti = iv.getValue()*sigmaV.get(idx);
                    sigmaV.increment(idx, c*x_ti*x_ti);
                }
                break;
            case PROJECT:
                for(IndexValue iv : x_t)//only the nonzero values in x_t will cause a change in value
                {
                    int idx = iv.getIndex();
                    double x_r = iv.getValue();
                    double S_rr = sigmaV.get(idx);
                    sigmaV.set(idx, 1/(1/S_rr+numer*x_r*x_r));
                }
                break;
            case EXACT:
                for(IndexValue iv : x_t)//only the nonzero values in x_t will cause a change in value
                {
                    int idx = iv.getIndex();
                    double x_r = iv.getValue();
                    double S_rr = sigmaV.get(idx);
                    sigmaV.set(idx, S_rr/(Math.pow(S_rr*x_r*x_r*C+1, 2)));
                }
                break;
        }

        //zero out temp space
        if(covMode == CovMode.FULL)
            Sigma_xt.zeroOut();
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
}
