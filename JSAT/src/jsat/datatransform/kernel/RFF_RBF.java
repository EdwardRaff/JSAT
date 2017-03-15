package jsat.datatransform.kernel;

import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransformBase;
import jsat.distributions.Distribution;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.RandomMatrix;
import jsat.linear.RandomVector;
import jsat.linear.Vec;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * An Implementation of Random Fourier Features for the {@link RBFKernel}. It 
 * transforms the numerical variables of a feature space to form a new feature 
 * space where the dot product between features approximates the RBF Kernel 
 * product. 
 * <br><br>
 * See: Rahimi, A.,&amp;Recht, B. (2007). <i>Random Features for Large-Scale 
 * Kernel Machines</i>. Neural Information Processing Systems. Retrieved from 
 * <a href="http://seattle.intel-research.net/pubs/rahimi-recht-random-features.pdf">
 * here</a>
 * @author Edward Raff
 */
public class RFF_RBF extends DataTransformBase
{

    private static final long serialVersionUID = -3478216020648280477L;
    private Matrix transform;
    private Vec offsets;
    private double sigma;
    private int dim;
    private boolean inMemory;
   
    /**
     * Creates a new RFF RBF object that will use an transformed feature space
     * with a dimensionality of 512. This constructor should be used with a
     * parameter search to find a good value for {@link #setSigma(double) sigma}
     */
    public RFF_RBF()
    {
        this(1.0);
    }
    
    /**
     * Creates a new RFF RBF object that will use an transformed feature space
     * with a dimensionality of 512.
     *
     * @param sigma the positive sigma value for the {@link RBFKernel}
     */
    public RFF_RBF(double sigma)
    {
        this(sigma, 512);
    }
    
    /**
     * Creates a new RFF RBF object
     *
     * @param sigma the positive sigma value for the {@link RBFKernel}
     * @param dim the new feature size dimension to project into.
     */
    public RFF_RBF(double sigma, int dim)
    {
        this(sigma, dim, true);
    }
    
    /**
     * Creates a new RFF RBF object
     *
     * @param sigma the positive sigma value for the {@link RBFKernel}
     * @param dim the new feature size dimension to project into.
     * @param inMemory {@code true} if the internal matrix should be stored in
     * memory. If {@code false}, the memory will be re-computed as needed,
     * increasing computation cost but uses no extra memory.
     */
    public RFF_RBF(double sigma, int dim, boolean inMemory)
    {
        setSigma(sigma);
        setDimensions(dim);
        setInMemory(inMemory);
    }

    /**
     * Creates a new RFF RBF object
     * @param featurSize the number of numeric features in the original feature 
     * space
     * @param sigma the positive sigma value for the {@link RBFKernel} 
     * @param dim the new feature size dimension to project into. 
     * @param rand the source of randomness to initialize internal state
     * @param inMemory {@code true} if the internal matrix should be stored in 
     * memory. If {@code false}, the memory will be re-computed as needed, 
     * increasing computation cost but uses no extra memory. 
     */
    public RFF_RBF(int featurSize, double sigma, int dim, Random rand, boolean inMemory)
    {
        this(sigma, dim, inMemory);
        if(featurSize <= 0)
            throw new IllegalArgumentException("The number of numeric features must be positive, not " + featurSize);
        if(sigma <= 0 || Double.isInfinite(sigma) || Double.isNaN(sigma))
            throw new IllegalArgumentException("The sigma parameter must be positive, not " + sigma);
        if(dim <= 1)
            throw new IllegalArgumentException("The target dimension must be positive, not " + dim);
        transform = new RandomMatrixRFF_RBF(Math.sqrt(0.5/(sigma*sigma)), featurSize, dim, rand.nextLong());
        offsets = new RandomVectorRFF_RBF(dim, rand.nextLong());
        
        if(inMemory)
        {
            transform = transform.add(0.0);//will copy into a new mutable and add nothing
            offsets = new DenseVector(offsets);
        }
    }

    @Override
    public void fit(DataSet data)
    {
        int featurSize = data.getNumNumericalVars();
        Random rand = RandomUtil.getRandom();
        transform = new RandomMatrixRFF_RBF(Math.sqrt(0.5/(sigma*sigma)), featurSize, dim, rand.nextLong());
        offsets = new RandomVectorRFF_RBF(dim, rand.nextLong());
        
        if(inMemory)
        {
            transform = transform.add(0.0);//will copy into a new mutable and add nothing
            offsets = new DenseVector(offsets);
        }
    }
    
    

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected RFF_RBF(RFF_RBF toCopy)
    {
        if(toCopy.transform != null)
            this.transform = toCopy.transform.clone();
        if(toCopy.offsets != null)
            this.offsets = toCopy.offsets.clone();
        this.dim = toCopy.dim;
        this.inMemory = toCopy.inMemory;
        this.sigma = toCopy.sigma;
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec oldX = dp.getNumericalValues();
        Vec newX = oldX.multiply(transform);
        
        final double coef = Math.sqrt(2.0/transform.cols());
        for(int i = 0; i < newX.length(); i++)
            newX.set(i, Math.cos(newX.get(i)+offsets.get(i))*coef);
        
        return new DataPoint(newX, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }

    @Override
    public RFF_RBF clone()
    {
        return new RFF_RBF(this);
    }
    
    private static class RandomMatrixRFF_RBF extends RandomMatrix
    {

        private static final long serialVersionUID = 4702514384718636893L;
        private double coef;

        public RandomMatrixRFF_RBF(double coef, int rows, int cols, long seedMult)
        {
            super(rows, cols, seedMult);
            this.coef = coef;
        }
        
        @Override
        protected double getVal(Random rand)
        {
            return coef*rand.nextGaussian();
        }
    }
    
    private static class RandomVectorRFF_RBF extends RandomVector
    {

        private static final long serialVersionUID = -6132378281909907937L;

        public RandomVectorRFF_RBF(int length, long seedMult)
        {
            super(length, seedMult);
        }
        
        @Override
        protected double getVal(Random rand)
        {
            return rand.nextDouble()*2*Math.PI;
        }

        @Override
        public Vec clone()
        {
            return this;
        }
        
    }

    
    /**
     * Sets whether or not the transform matrix is stored explicitly in memory
     * or not. Explicit storage is often faster, but can be prohibitive for
     * large feature sizes
     * @param inMemory {@code true} to explicitly store the transform matrix,
     * {@code false} to re-create it on the fly as needed
     */
    public void setInMemory(boolean inMemory)
    {
        this.inMemory = inMemory;
    }

    /**
     * 
     * @return {@code true} if this object will explicitly store the transform
     * matrix, {@code false} to re-create it on the fly as needed
     */
    public boolean isInMemory()
    {
        return inMemory;
    }
    
    /**
     * Sets the number of dimensions in the new approximate space to use. This
     * will be the number of numeric features in the transformed data, and
     * larger values increase the accuracy of the approximation.
     *
     * @param dimensions
     */
    public void setDimensions(int dimensions)
    {
        if (dimensions < 1)
            throw new ArithmeticException("Number of dimensions must be a positive value, not " + dimensions);
        this.dim = dimensions;
    }

    /**
     * Returns the number of dimensions that will be used in the projected space
     *
     * @return the number of dimensions that will be used in the projected space
     */
    public int getDimensions()
    {
        return dim;
    }

    /**
     * Sets the &sigma; parameter of the RBF kernel that is being approximated.
     *
     * @param sigma the positive value to use for &sigma;
     * @see RBFKernel#setSigma(double)
     */
    public void setSigma(double sigma)
    {
        if (sigma <= 0.0 || Double.isInfinite(sigma) || Double.isNaN(sigma))
            throw new IllegalArgumentException("Sigma must be a positive value, not " + sigma);
        this.sigma = sigma;
    }

    /**
     * Returns the &sigma; value used for the RBF kernel approximation.
     *
     * @return the &sigma; value used for the RBF kernel approximation.
     */
    public double getSigma()
    {
        return sigma;
    }

    /**
     * Guess the distribution to use for the kernel width term
     * {@link #setSigma(double) &sigma;} in the RBF kernel being approximated.
     *
     * @param d the data set to get the guess for
     * @return the guess for the &sigma; parameter in the RBF Kernel
     */
    public Distribution guessSigma(DataSet d)
    {
        return RBFKernel.guessSigma(d);
    }
}
