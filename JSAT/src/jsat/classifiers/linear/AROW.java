package jsat.classifiers.linear;

import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.MathTricks;

/**
 * An implementation of Adaptive Regularization of Weight Vectors (AROW), which 
 * uses second order information to learn a large margin binary classifier. As 
 * such, updates can occur on correctly classified instances if they are not far
 * enough from the margin. Unlike many margin algorithms, it handles noise well.
 * <br>
 * NOTE: This implementation does not add an implicit bias term, so the solution
 * goes through the origin 
 * <br><br>
 * See: Crammer, K., Kulesza, A., & Dredze, M. (2013). <i>Adaptive 
 * regularization of weight vectors</i>. Machine Learning, 91(2), 155–187. 
 * doi:10.1007/s10994-013-5327-x
 * 
 * @author Edward Raff
 */
public class AROW extends BaseUpdateableClassifier
{
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
        
        if(diagonalOnly)
        {
            /* for the diagonal, its a pairwise multiplication. So just copy 
             * then multiply by the sigmas, ordes dosnt matter
             */
            if(x_t.isSparse())
            {
                //Faster to set only the needed final values
                for(IndexValue iv : x_t)
                    Sigma_xt.set(iv.getIndex(), iv.getValue()*sigmaV.get(iv.getIndex()));
            }
            else
            {
                x_t.copyTo(Sigma_xt);
                Sigma_xt.mutablePairwiseMultiply(sigmaV);
            }
        }
        else
        {
            sigmaM.multiply(x_t, 1, Sigma_xt);
        }
        
        double v_t = x_t.dot(Sigma_xt);
        double b_t_inv = v_t+r;
        
        double alpha_t = Math.max(0, 1-y_t*m_t)/b_t_inv;
        w.mutableAdd(alpha_t*y_t, Sigma_xt);
        
        if(diagonalOnly)
        {
            /* diagonal is pairwise products as well:
             * S += S x x' S
             * S x == x' S b/c symmetry
             * S += Sx Sx
             * so just square the values and then add 
             */
            Sigma_xt.applyFunction(MathTricks.sqrdFunc);
            sigmaV.mutableAdd(-1/b_t_inv, Sigma_xt);
        }
        else
        {
            //Because Sigma is symetric, x*S == S*x
            Matrix.OuterProductUpdate(sigmaM, Sigma_xt, Sigma_xt, -1/b_t_inv);
        }
        
        //Zero out temp store
        if(diagonalOnly && x_t.isSparse())//only these values will be non zero 
            for(IndexValue iv : x_t)
                Sigma_xt.set(iv.getIndex(), 0.0);
        else
            Sigma_xt.zeroOut();
        
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(w == null)
            throw new UntrainedModelException("Model has not yet ben trained");
        CategoricalResults cr = new CategoricalResults(2);
        double score = w.dot(data.getNumericalValues());
        if(score < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
}
