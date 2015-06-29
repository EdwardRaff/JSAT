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
 * An implementation of Adaptive Regularization of Weight Vectors (AROW), which 
 * uses second order information to learn a large margin binary classifier. As 
 * such, updates can occur on correctly classified instances if they are not far
 * enough from the margin. Unlike many margin algorithms, it handles noise well.
 * <br>
 * NOTE: This implementation does not add an implicit bias term, so the solution
 * goes through the origin 
 * <br><br>
 * See: Crammer, K., Kulesza, A.,&amp;Dredze, M. (2013). <i>Adaptive 
 * regularization of weight vectors</i>. Machine Learning, 91(2), 155–187. 
 * doi:10.1007/s10994-013-5327-x
 * 
 * @author Edward Raff
 */
public class AROW extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized, SingleWeightVectorModel
{

    private static final long serialVersionUID = 443803827811508204L;
    private Vec w;
    /**
     * Full covariance matrix
     */
    private Matrix sigmaM;
    /**
     * Diagonal only covariance matrix
     */
    private Vec sigmaV;
    private boolean diagonalOnly = false;
    private double r;
    
    /**
     * Temp vector used to store Sigma * x_t. Make sure the vector is zeroed out
     * before returning from update
     */
    private Vec Sigma_xt;

    /**
     * Creates a new AROW learner
     */
    public AROW()
    {
        this(1e-2, true);
    }

    /**
     * Creates a new AROW learner
     * @param r the regularization parameter
     * @param diagonalOnly whether or not to use only the diagonal of the covariance 
     * @see #setR(double) 
     * @see #setDiagonalOnly(boolean) 
     */
    public AROW(double r, boolean diagonalOnly)
    {
        setR(r);
        setDiagonalOnly(diagonalOnly);
    }

    /**
     * Copy constructor
     * @param other object to copy
     */
    protected AROW(AROW other)
    {
        this.r = other.r;
        this.diagonalOnly = other.diagonalOnly;
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
     * Using the full covariance matrix requires <i>O(d<sup>2</sup>)</i> work on 
     * mistakes, where <i>d</i> is the dimension of the data. Runtime can be 
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
     * Sets the r parameter of AROW, which controls the regularization. Larger 
     * values reduce the change in the model on each update. 
     * @param r the regularization parameter in (0, Inf)
     */
    public void setR(double r)
    {
        if(Double.isNaN(r) || Double.isInfinite(r) || r <= 0)
            throw new IllegalArgumentException("r must be a postive constant, not " + r);
        this.r = r;
    }

    /**
     * Returns the regularization parameter
     * @return the regularization parameter
     */
    public double getR()
    {
        return r;
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
    public AROW clone()
    {
        return new AROW(this);
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
        
        double m_t = x_t.dot(w);
        if(y_t == Math.signum(m_t))
            return;//no update needed
        
        double v_t = 0;
        
        if(diagonalOnly)
        {
            for(IndexValue iv : x_t)
            {
                double x_ti = iv.getValue();
                v_t += x_ti * x_ti * sigmaV.get(iv.getIndex());
            }
        }
        else
        {
            sigmaM.multiply(x_t, 1, Sigma_xt);
            v_t = x_t.dot(Sigma_xt);
        }
        
        double b_t_inv = v_t+r;
        
        double alpha_t = Math.max(0, 1-y_t*m_t)/b_t_inv;
        if(!diagonalOnly)
            w.mutableAdd(alpha_t * y_t, Sigma_xt);
        else
            for (IndexValue iv : x_t)
                w.increment(iv.getIndex(), alpha_t * y_t * iv.getValue() * sigmaV.get(iv.getIndex()));

        if(diagonalOnly)
        {
            /* diagonal is pairwise products as well:
             * S += S x x' S
             * S x == x' S b/c symmetry
             * S += Sx Sx
             * so just square the values and then add 
             */
            for(IndexValue iv : x_t)
            {
                int idx = iv.getIndex();
                double xt_i = iv.getValue()*sigmaV.get(idx);
                sigmaV.increment(idx, -(xt_i*xt_i)/b_t_inv);
            }
        }
        else
        {
            //Because Sigma is symetric, x*S == S*x
            Matrix.OuterProductUpdate(sigmaM, Sigma_xt, Sigma_xt, -1/b_t_inv);
        }
        
        //Zero out temp store
        if(diagonalOnly)
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
    
    /**
     * Guess the distribution to use for the regularization term
     * {@link #setR(double) r} .
     *
     * @param d the data set to get the guess for
     * @return the guess for the r parameter
     */
    public static Distribution guessR(DataSet d)
    {
        return new LogUniform(Math.pow(2, -4), Math.pow(2, 4));//from Exact Soft Confidence-Weighted Learning paper
    }
}
