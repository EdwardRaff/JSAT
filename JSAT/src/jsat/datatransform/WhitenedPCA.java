package jsat.datatransform;

import java.util.Comparator;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.*;
import static jsat.linear.MatrixStatistics.*;

/**
 * An extension of {@link PCA} that attempts to capture the variance, and make 
 * the variables in the output space independent from each-other. An of equal 
 * scale, so that the covariance is equal to {@link Matrix#eye(int) I}. The 
 * results may be further from the identity matrix than desired as the target 
 * dimension shrinks<br>
 * <br>
 * The Whitened PCA is more computational expensive than the normal PCA 
 * algorithm, but transforming the data takes the same time. 
 * 
 * @author Edward Raff
 */
public class WhitenedPCA implements DataTransform
{
    /**
     * Regularization parameter
     */
    protected double regularization;
    /**
     * The number of dimensions to project down to
     */
    protected int dims;
    
    /**
     * The final transformation matrix, that will create new points 
     * <tt>y</tt> = <tt>transform</tt> * x
     */
    protected Matrix transform;

    /**
     * Creates a new WhitenedPCA
     * @param dataSet the data set to whiten
     * @param regularization the amount of regularization to add, avoids numerical instability
     * @param dims the number of dimensions to project down to
     */
    public WhitenedPCA(DataSet dataSet, double regularization, int dims)
    {
        setRegularization(regularization);
        setDims(dims);
        
        setUpTransform(getSVD(dataSet));
        
    }

    /**
     * Creates a new WhitenedPCA, the dimensions will be chosen so that the 
     * subset of dimensions is of full rank. 
     * 
     * @param dataSet the data set to whiten
     * @param regularization the amount of regularization to add, avoids numerical instability
     */
    public WhitenedPCA(DataSet dataSet, double regularization)
    {
        setRegularization(regularization);
        SingularValueDecomposition svd = getSVD(dataSet);
        setDims(svd.getRank());
        
        
        setUpTransform(svd);
    }
    
    /**
     * Creates a new WhitenedPCA. The dimensions will be chosen so that the 
     * subset of dimensions is of full rank. The regularization parameter will be
     * chosen as the log of the condition of the covariance. 
     * 
     * @param dataSet the data set to whiten
     */
    public WhitenedPCA(DataSet dataSet)
    {
        
        SingularValueDecomposition svd = getSVD(dataSet);
        setRegularization(svd);
        setDims(svd.getRank());
        
        
        setUpTransform(svd);
    }
    
    /**
     * Creates a new WhitenedPCA. The regularization parameter will be
     * chosen as the log of the condition of the covariance. 
     * 
     * @param dataSet the data set to whiten
     * @param dims the number of dimensions to project down to
     */
    public WhitenedPCA(DataSet dataSet, int dims)
    {
        
        SingularValueDecomposition svd = getSVD(dataSet);
        setRegularization(svd);
        setDims(dims);
        
        
        setUpTransform(svd);
    }
    
    /**
     * Copy constructor 
     * @param other the transform to make a copy of
     */
    private WhitenedPCA(WhitenedPCA other)
    {
        this.regularization = other.regularization;
        this.dims = other.dims;
        this.transform = other.transform.clone();
    }

    /**
     * Gets a SVD for the covariance matrix of the data set
     * @param dataSet the data set in question
     * @return the SVD for the covariance
     */
    private SingularValueDecomposition getSVD(DataSet dataSet)
    {
        Matrix cov = covarianceMatrix(meanVector(dataSet), dataSet);
        for(int i = 0; i < cov.rows(); i++)//force it to be symmetric
            for(int j = 0; j < i; j++)
                cov.set(j, i, cov.get(i, j));
        EigenValueDecomposition evd = new EigenValueDecomposition(cov);
        //Sort form largest to smallest
        evd.sortByEigenValue(new Comparator<Double>() 
        {
            @Override
            public int compare(Double o1, Double o2)
            {
                return -Double.compare(o1, o2);
            }
        });
        return new SingularValueDecomposition(evd.getVRaw(), evd.getVRaw(), evd.getRealEigenvalues());
    }
    

    /**
     * Creates the {@link #transform transform matrix} to be used when 
     * converting data points. It is called in the constructor after all values
     * are set. 
     * 
     * @param svd the SVD of the covariance of the source data set
     */
    protected void setUpTransform(SingularValueDecomposition svd)
    {
        Vec diag = new DenseVector(dims);
        
        double[] s = svd.getSingularValues();
        
        for(int i = 0; i < dims; i++)
            diag.set(i, 1.0/Math.sqrt(s[i]+regularization));
        
        transform = new SubMatrix(svd.getU().transpose(), 0, 0, dims, s.length).clone();
        
        Matrix.diagMult(diag, transform);
    }
    

    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec newVec = transform.multiply(dp.getNumericalValues());
        
        DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
        
        return newDp;
    }
    
    
    private void setRegularization(double regularization)
    {
        if(regularization < 0 || Double.isNaN(regularization) || Double.isInfinite(regularization))
            throw new ArithmeticException("Regularization must be non negative value, not " + regularization);
        this.regularization = regularization;
    }

    private void setDims(int dims)
    {
        if(dims < 1)
            throw new ArithmeticException("Invalid number of dimensions, bust be > 0");
        this.dims = dims;
    }

    @Override
    public DataTransform clone()
    {
        return new WhitenedPCA(this);
    }

    private void setRegularization(SingularValueDecomposition svd)
    {
        if(svd.isFullRank())
            setRegularization(1e-10);
        else
            setRegularization(Math.max(Math.log(1.0+svd.getSingularValues()[svd.getRank()])*0.25, 1e-4));
    }
    
    static public class WhitenedPCATransformFactory implements DataTransformFactory
    {
        private int maxPCs;

        /**
         * Creates a new WhitenedPCA Factory
         * @param maxPCs the number of principle components
         */
        public WhitenedPCATransformFactory(int maxPCs)
        {
            this.maxPCs = maxPCs;
        }
        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new WhitenedPCA(dataset, maxPCs);
        }
        
    }
}
