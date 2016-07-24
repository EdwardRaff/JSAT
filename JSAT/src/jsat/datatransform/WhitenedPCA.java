package jsat.datatransform;

import java.util.Comparator;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.UniformDiscrete;
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
public class WhitenedPCA extends DataTransformBase
{

    private static final long serialVersionUID = 6134243673037330608L;
    /**
     * Regularization parameter
     */
    protected double regularization;
    /**
     * The number of dimensions to project down to
     */
    protected int dimensions;
    
    /**
     * The final transformation matrix, that will create new points 
     * <tt>y</tt> = <tt>transform</tt> * x
     */
    protected Matrix transform;

    /**
     * Creates a new WhitenedPCA transform that uses up to 50 dimensions for the
     * transformed space. This may not be optimal for any given dataset.
     *
     * @param dims the number of dimensions to project down to
     */
    public WhitenedPCA()
    {
        this(50);
    }

    /**
     * Creates a new WhitenedPCA transform
     *
     * @param dims the number of dimensions to project down to
     */
    public WhitenedPCA(int dims)
    {
        this(1e-4, dims);
    }

    /**
     * Creates a new WhitenedPCA transform
     *
     * @param regularization the amount of regularization to add, avoids
     * numerical instability
     * @param dims the number of dimensions to project down to
     */
    public WhitenedPCA(double regularization, int dims)
    {
        setRegularization(regularization);
        setDimensions(dims);
    }
            
    /**
     * Creates a new WhitenedPCA from the given dataset
     * @param dataSet the data set to whiten
     * @param regularization the amount of regularization to add, avoids numerical instability
     * @param dims the number of dimensions to project down to
     */
    public WhitenedPCA(DataSet dataSet, double regularization, int dims)
    {
        this(regularization, dims);
        fit(dataSet);
    }

    @Override
    public void fit(DataSet dataSet)
    {
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
        setDimensions(svd.getRank());
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
        setDimensions(svd.getRank());
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
        setDimensions(dims);
        
        
        setUpTransform(svd);
    }
    
    /**
     * Copy constructor 
     * @param other the transform to make a copy of
     */
    private WhitenedPCA(WhitenedPCA other)
    {
        this.regularization = other.regularization;
        this.dimensions = other.dimensions;
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
        Vec diag = new DenseVector(dimensions);
        
        double[] s = svd.getSingularValues();
        
        for(int i = 0; i < dimensions; i++)
            diag.set(i, 1.0/Math.sqrt(s[i]+regularization));
        
        transform = new SubMatrix(svd.getU().transpose(), 0, 0, dimensions, s.length).clone();
        
        Matrix.diagMult(diag, transform);
    }
    

    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec newVec = transform.multiply(dp.getNumericalValues());
        
        DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
        
        return newDp;
    }
    
    /**
     * 
     * @param regularization the regularization to apply to the diagonal of the
     * decomposition. This can improve numeric stability and reduces noise.
     */
    public void setRegularization(double regularization)
    {
        if(regularization < 0 || Double.isNaN(regularization) || Double.isInfinite(regularization))
            throw new ArithmeticException("Regularization must be non negative value, not " + regularization);
        this.regularization = regularization;
    }

    /**
     * 
     * @return the amount of regularization to apply
     */
    public double getRegularization()
    {
        return regularization;
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
    
    /**
     * Sets the number of dimensions to project down to
     *
     * @param dimensions the feature size to project down to
     */
    public void setDimensions(int dimensions)
    {
        if (dimensions < 1)
            throw new IllegalArgumentException("Number of dimensions must be positive, not " + dimensions);
        this.dimensions = dimensions;
    }

    /**
     * Returns the number of dimensions to project down to
     *
     * @return the number of dimensions to project down to
     */
    public int getDimensions()
    {
        return dimensions;
    }
    
    public static Distribution guessDimensions(DataSet d)
    {
        //TODO improve using SVD rank
        if(d.getNumNumericalVars() < 100)
            return new UniformDiscrete(1, d.getNumNumericalVars());
        return new UniformDiscrete(20, 100);
    }
}
