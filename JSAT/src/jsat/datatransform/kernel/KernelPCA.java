package jsat.datatransform.kernel;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransformBase;
import jsat.datatransform.PCA;
import jsat.datatransform.kernel.Nystrom.SamplingMethod;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.EigenValueDecomposition;
import jsat.linear.Matrix;
import jsat.linear.RowColumnOps;
import jsat.linear.Vec;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.utils.random.RandomUtil;

/**
 * A kernelized implementation of {@link PCA}. Because this works in a different
 * feature space, it will do its own centering in the kernel space. 
 * <br><br>
 * KernelPCA is expensive to compute at O(n<sup>3</sup>) work, where <i>n</i> is
 * the number of data points. For this reason, sampling from {@link Nystrom} is
 * used to reduce the data set to a reasonable approximation. 
 * <br><br>
 * See: Schölkopf, B., Smola, A.,&amp;Müller, K.-R. (1998). <i>Nonlinear Component
 * Analysis as a Kernel Eigenvalue Problem</i>. Neural Computation, 10(5), 
 * 1299–1319. doi:10.1162/089976698300017467
 * 
 * @author Edward Raff
 * @see Nystrom.SamplingMethod
 */
public class KernelPCA extends DataTransformBase
{

    private static final long serialVersionUID = 5676602024560381023L;

    /**
     * The dimension to project down to
     */
    private int dimensions;
    @ParameterHolder
    private KernelTrick k;
    private int basisSize;
    private Nystrom.SamplingMethod samplingMethod;
    
    private double[] eigenVals;
    /**
     * The matrix of transformed eigen vectors
     */
    private Matrix eigenVecs;
    /**
     * The vecs used for the transform
     */
    private Vec[] vecs;
    
    //row / colum info for centering in the feature space
    private double[] rowAvg;
    private double allAvg;
    
    /**
     * Creates a new Kernel PCA transform object using the
     * {@link RBFKernel RBF Kernel} and 100 dimensions
     *
     */
    public KernelPCA()
    {
        this(100);
    }
    
    /**
     * Creates a new Kernel PCA transform object using the
     * {@link RBFKernel RBF Kernel}
     *
     * @param dimensions the number of dimensions to project down to. Must be
     * less than than the basis size
     */
    public KernelPCA(int dimensions)
    {
        this(new RBFKernel(), dimensions);
    }

    /**
     * Creates a new Kernel PCA transform object
     *
     * @param k the kernel trick to use
     * @param dimensions the number of dimensions to project down to. Must be
     * less than than the basis size
     */
    public KernelPCA(KernelTrick k, int dimensions)
    {
        this(k, dimensions, 1000, SamplingMethod.UNIFORM);
    }

    /**
     * Creates a new Kernel PCA transform object
     * @param k the kernel trick to use
     * @param dimensions the number of dimensions to project down to. Must be 
     * less than than the basis size
     * @param basisSize the number of points from the data set to select. If
     * larger than the number of data points in the data set, the whole data set
     * will be used. 
     * @param samplingMethod the sampling method to select the basis vectors
     */
    public KernelPCA(KernelTrick k, int dimensions, int basisSize, Nystrom.SamplingMethod samplingMethod)
    {
        setDimensions(dimensions);
        setKernel(k);
        setBasisSize(basisSize);
        setBasisSamplingMethod(samplingMethod);
    }
    
    /**
     * Creates a new Kernel PCA transform object
     * @param k the kernel trick to use
     * @param ds the data set to form the data transform from
     * @param dimensions the number of dimensions to project down to. Must be 
     * less than than the basis size
     * @param basisSize the number of points from the data set to select. If
     * larger than the number of data points in the data set, the whole data set
     * will be used. 
     * @param samplingMethod the sampling method to select the basis vectors
     */
    public KernelPCA(KernelTrick k, DataSet ds, int dimensions, int basisSize, Nystrom.SamplingMethod samplingMethod)
    {
        this(k, dimensions, basisSize, samplingMethod);
        fit(ds);
    }

    @Override
    public void fit(DataSet ds)
    {
        if(ds.getSampleSize() <= basisSize)
        {
            vecs = new Vec[ds.getSampleSize()];
            for(int i = 0; i < vecs.length; i++)
                vecs[i] = ds.getDataPoint(i).getNumericalValues();
        }
        else
        {
            int i = 0;
            List<Vec> sample = Nystrom.sampleBasisVectors(k, ds, ds.getDataVectors(), samplingMethod, basisSize, false, RandomUtil.getRandom());
            vecs = new Vec[sample.size()];
            for(Vec v : sample)
                vecs[i++] = v;
        }
        Matrix K = new DenseMatrix(vecs.length, vecs.length);
        
        //Info used to compute centered Kernel matrix
        rowAvg = new double[K.rows()];
        allAvg = 0;
        
        for(int i = 0; i < K.rows(); i++)
        {
            Vec x_i = vecs[i];
            for(int j = i; j < K.cols(); j++)
            {
                double K_ij = k.eval(x_i, vecs[j]);
                K.set(i, j, K_ij);

                K.set(j, i, K_ij);//K = K'
            }
        }
        
        //Get row / col info to perform centering. Since K is symetric, the row 
        //and col info are the same
        for(int i = 0; i < K.rows(); i++)
            for(int j = 0; j < K.cols(); j++)
                rowAvg[i] += K.get(i, j);
        
        for (int i = 0; i < K.rows(); i++)
        {
            allAvg += rowAvg[i];
            rowAvg[i] /= K.rows();
        }
        
        allAvg /= (K.rows()*K.cols());

        
        //Centered version of the marix
        //K_c(i, j) = K_ij - sum_z K_zj / m - sum_z K_iz / m + sum_{z,y} K_zy / m^2
        
        for(int i = 0; i < K.rows(); i++)
            for(int j = 0; j < K.cols(); j++)
                K.set(i, j, K.get(i, j) - rowAvg[i] - rowAvg[j] + allAvg);
        
        
        EigenValueDecomposition evd = new EigenValueDecomposition(K);
        evd.sortByEigenValue(new Comparator<Double>() 
        {
            @Override
            public int compare(Double o1, Double o2)
            {
                return -Double.compare(o1, o2);
            }
        });
        
        eigenVals = evd.getRealEigenvalues();
        eigenVecs = evd.getV();
        for(int j = 0; j < eigenVals.length; j++)//TODO row order would be more cache friendly 
            RowColumnOps.divCol(eigenVecs, j, Math.sqrt(eigenVals[j]));
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected KernelPCA(KernelPCA toCopy)
    {
        this.dimensions = toCopy.dimensions;
        this.k = toCopy.k.clone();
        this.basisSize = toCopy.basisSize;
        this.samplingMethod = toCopy.samplingMethod;
        if(toCopy.eigenVals != null)
            this.eigenVals = Arrays.copyOf(toCopy.eigenVals, toCopy.eigenVals.length);
        if(toCopy.eigenVecs != null)
            this.eigenVecs = toCopy.eigenVecs.clone();
        if(toCopy.vecs != null)
        {
            this.vecs = new Vec[toCopy.vecs.length];
            for(int i = 0; i < vecs.length; i++)
                this.vecs[i] = toCopy.vecs[i].clone();
            this.rowAvg = Arrays.copyOf(toCopy.rowAvg, toCopy.rowAvg.length);
        }
        this.allAvg = toCopy.allAvg;
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec oldVec = dp.getNumericalValues();
        Vec newVec = new DenseVector(dimensions);
        
        //TODO put this in a thread local object? Or hope JVM puts a large array on the stack? 
        final double[] kEvals = new double[vecs.length];

        double tAvg = 0;

        for (int j = 0; j < vecs.length; j++)
            tAvg += (kEvals[j] = k.eval(vecs[j], oldVec));

        tAvg /= vecs.length;

        for (int i = 0; i < dimensions; i++)
        {
            double val = 0;
            for (int j = 0; j < vecs.length; j++)
                val += eigenVecs.get(j, i) * (kEvals[j] - tAvg - rowAvg[i] + allAvg);
            newVec.set(i, val);
        }

        return new DataPoint(newVec, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }

    @Override
    public KernelPCA clone()
    {
        return new KernelPCA(this);
    }

    /**
     * 
     * @param k the kernel trick to use
     */
    public void setKernel(KernelTrick k)
    {
        this.k = k;
    }

    /**
     * 
     * @return the kernel trick to use
     */
    public KernelTrick getKernel()
    {
        return k;
    }
    
    /**
     * Sets the basis size for the Kernel PCA to be learned from. Increasing the
     * basis increase the accuracy of the transform, but increased the training
     * time at a cubic rate.
     *
     * @param basisSize the number of basis vectors to build Kernel PCA from
     */
    public void setBasisSize(int basisSize)
    {
        if (basisSize < 1)
            throw new IllegalArgumentException("The basis size must be positive, not " + basisSize);
        this.basisSize = basisSize;
    }

    /**
     * Returns the number of basis vectors to use
     *
     * @return the number of basis vectors to use
     */
    public int getBasisSize()
    {
        return basisSize;
    }

    /**
     * Sets the dimension of the new feature space, which is the number of
     * principal components to select from the kernelized feature space.
     *
     * @param dimensions the number of dimensions to project down too
     */
    public void setDimensions(int dimensions)
    {
        if (dimensions < 1)
            throw new IllegalArgumentException("The number of dimensions must be positive, not " + dimensions);
        this.dimensions = dimensions;
    }

    /**
     * Returns the number of dimensions to project down too
     *
     * @return the number of dimensions to project down too
     */
    public int getDimensions()
    {
        return dimensions;
    }

    /**
     * Sets the method of selecting the basis vectors
     *
     * @param method the method of selecting the basis vectors
     */
    public void setBasisSamplingMethod(SamplingMethod method)
    {
        this.samplingMethod = method;
    }

    /**
     * Returns the method of selecting the basis vectors
     *
     * @return the method of selecting the basis vectors
     */
    public SamplingMethod getBasisSamplingMethod()
    {
        return samplingMethod;
    }
    
    public static Distribution guessDimensions(DataSet d)
    {
        return new UniformDiscrete(20, 200);
    }
}
