package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.*;

/**
 * An extension of {@link WhitenedPCA}, is the Whitened Zero Component Analysis.
 * Whitened ZCA can not project to a lower dimension, as it rotates the output 
 * in the original dimension.
 * 
 * @author Edward Raff
 */
public class WhitenedZCA extends WhitenedPCA implements InPlaceTransform
{

    private static final long serialVersionUID = 7546033727733619587L;
    private ThreadLocal<Vec> tempVecs;
    
    /**
     * Creates a new WhitenedZCA transform that uses up to 50 dimensions for the
     * transformed space. This may not be optimal for any given dataset.
     *
     * @param dims the number of dimensions to project down to
     */
    public WhitenedZCA()
    {
        this(50);
    }

    /**
     * Creates a new WhitenedZCA transform
     *
     * @param dims the number of dimensions to project down to
     */
    public WhitenedZCA(int dims)
    {
        this(1e-4, dims);
    }

    /**
     * Creates a new WhitenedZCA transform
     *
     * @param regularization the amount of regularization to add, avoids
     * numerical instability
     * @param dims the number of dimensions to project down to
     */
    public WhitenedZCA(double regularization, int dims)
    {
        setRegularization(regularization);
        setDimensions(dims);
    }

    /**
     * Creates a new Whitened ZCA transform from the given data.
     * 
     * @param dataSet the data set to whiten
     * @param regularization the amount of regularization to add, avoids 
     * numerical instability
     */
    public WhitenedZCA(DataSet dataSet, double regularization)
    {
        super(dataSet, regularization);
    }

    /**
     * Creates a new Whitened ZCA transform from the given data. The
     * regularization parameter will be chosen as the log of the condition of
     * the covariance.
     *
     * @param dataSet the data set to whiten
     */
    public WhitenedZCA(DataSet dataSet)
    {
        super(dataSet);
    }

    @Override
    public void fit(DataSet dataSet)
    {
        super.fit(dataSet);
        tempVecs = getThreadLocal(dataSet.getNumNumericalVars());
    }
    
    

    @Override
    public void mutableTransform(DataPoint dp)
    {
        Vec target = tempVecs.get();
        target.zeroOut();
        transform.multiply(dp.getNumericalValues(), 1.0, target);
        target.copyTo(dp.getNumericalValues());
    }

    @Override
    public boolean mutatesNominal()
    {
        return false;
    }

    @Override
    protected void setUpTransform(SingularValueDecomposition svd)
    {
        double[] s = svd.getSingularValues();
        Vec diag = new DenseVector(s.length);

        for(int i = 0; i < s.length; i++)
            diag.set(i, 1.0/Math.sqrt(s[i]+regularization));
        
        Matrix U = svd.getU();
        
        transform = U.multiply(Matrix.diag(diag)).multiply(U.transpose());
    }

    private ThreadLocal<Vec> getThreadLocal(final int dim)
    {
        return new ThreadLocal<Vec>()
        {

            @Override
            protected Vec initialValue()
            {
                return new DenseVector(dim);
            }
        };
    }
}
