package jsat.datatransform.kernel;

import java.util.*;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.clustering.SeedSelectionMethods;
import jsat.datatransform.*;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.*;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.utils.DoubleList;
import jsat.utils.IntSet;
import jsat.utils.random.RandomUtil;

/**
 * An implementation of the Nystrom approximation for any Kernel Trick. The full
 * rank kernel is approximated by a basis set of a subset of the data points 
 * that make up the original data set. Instead of explicitly forming the 
 * approximately decomposed matrix, this transform projects the original numeric
 * features of a data set into a new feature space where the dot product in the 
 * linear space approximates the dot product in the given kernel space. 
 * <br><br>
 * See: <br>
 * <ul>
 * <li>Williams, C.,&amp;Seeger, M. (2001). <i>Using the Nyström Method to Speed 
 * Up Kernel Machines</i>. Advances in Neural Information Processing Systems 13 
 * (pp. 682–688). MIT Press. Retrieved from 
 * <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.18.7519">
 * here</a></li>
 * <li>Yang, T., Li, Y.-F., Mahdavi, M., Jin, R.,&amp;Zhou, Z.-H. (2012). <i>
 * Nystrom Method vs Random Fourier Features A Theoretical and Empirical 
 * Comparison</i>. Advances in Neural Information Processing Systems 
 * (pp. 485–493). Retrieved from 
 * <a href="http://books.nips.cc/papers/files/nips25/NIPS2012_0248.txt">here</a>
 * </li>
 * <li>Kumar, S., Mohri, M.,&amp;Talwalkar, A. (2012). <i>Sampling methods for the
 * Nyström method</i>. The Journal of Machine Learning Research, 5, 981–1006. 
 * Retrieved from <a href="http://dl.acm.org/citation.cfm?id=2343678">here</a>
 * </li>
 * </ul>
 * @author Edward Raff
 */
public class Nystrom extends DataTransformBase
{

    private static final long serialVersionUID = -3227844260130709773L;
    private double ridge;
    @ParameterHolder
    private KernelTrick k;
    private int dimension;
    private SamplingMethod method;
    int basisSize;
    private boolean sampleWithReplacment;

    private List<Vec> basisVecs;
    private List<Double> accelCache;
    private Matrix transform;

    /**
     * Different sample methods may be used to select a better and more 
     * representative set of vectors to form the basis vectors at increased 
     * cost, where {@code n} is the number of data points in the full data set 
     * and {@code b} is the number of basis vectors to obtain. 
     */
    public enum SamplingMethod
    {
        /**
         * Selects the basis vectors by uniform sampling, takes O(b) time
         */
        UNIFORM, 
        /**
         * Selects the basis vectors by a weighted probability of the kernel 
         * value of k(x<sub>i</sub>, x<sub>i</sub>) for each <i>i</i>. If a 
         * kernel returns 1 for all k(x<sub>i</sub>, x<sub>i</sub>), then this 
         * reduces the uniform sampling. Takes O(n) time
         */
        DIAGONAL,
        /**
         * Selects the basis vectors by the weighted probability of the column 
         * norms of the gram matrix for each vector. Takes O(n<sup>2</sup>) time
         */
        NORM,
        /**
         * Selects the basis vectors as the means of a k-means clustering. Takes
         * the time needed to perform k-means
         */
        KMEANS,
    }
    
    /**
     * Creates a new Nystrom approximation object
     * @param k the kernel trick to form an approximation of
     * @param dataset the data set to form the approximate feature space from
     * @param basisSize the number of basis vectors to use, this is the output 
     * dimension size.
     * @param method what sampling method should be used to select the basis 
     * vectors from the full data set. 
     */
    public Nystrom(KernelTrick k, DataSet dataset, int basisSize, SamplingMethod method)
    {
        this(k, dataset, basisSize, method, 0.0, false);
    }

    /**
     * Creates a new Nystrom approximation object using the
     * {@link RBFKernel RBF Kernel} with 500 basis vectors
     *
     */
    public Nystrom()
    {
        this(new RBFKernel(), 500);
    }
    
    /**
     * Creates a new Nystrom approximation object
     *
     * @param k the kernel trick to form an approximation of
     * @param basisSize the number of basis vectors to use, this is the output
     * dimension size.
     */
    public Nystrom(KernelTrick k, int basisSize)
    {
        this(k, basisSize, SamplingMethod.UNIFORM, 1e-5, false);
    }
    
    /**
     * Creates a new Nystrom approximation object
     * @param k the kernel trick to form an approximation of
     * @param basisSize the number of basis vectors to use, this is the output 
     * dimension size.
     * @param method what sampling method should be used to select the basis 
     * vectors from the full data set. 
     * @param ridge a non negative additive term to regularize the eigen values 
     * of the decomposition. 
     * @param sampleWithReplacment {@code true} if the basis vectors should be 
     * sampled with replacement, {@code false} if they should not. 
     */
    public Nystrom(KernelTrick k, int basisSize, SamplingMethod method, double ridge, boolean sampleWithReplacment )
    {
        setKernel(k);
        setBasisSize(basisSize);
        setBasisSamplingMethod(method);
        setRidge(ridge);
        this.sampleWithReplacment = sampleWithReplacment;
    }
    
    /**
     * Creates a new Nystrom approximation object
     * @param k the kernel trick to form an approximation of
     * @param dataset the data set to form the approximate feature space from
     * @param basisSize the number of basis vectors to use, this is the output 
     * dimension size.
     * @param method what sampling method should be used to select the basis 
     * vectors from the full data set. 
     * @param ridge a non negative additive term to regularize the eigen values 
     * of the decomposition. 
     * @param sampleWithReplacment {@code true} if the basis vectors should be 
     * sampled with replacement, {@code false} if they should not. 
     */
    @SuppressWarnings("fallthrough")
    public Nystrom(KernelTrick k, DataSet dataset, int basisSize, SamplingMethod method, double ridge, boolean sampleWithReplacment )
    {
        this(k, basisSize, method, ridge, sampleWithReplacment);
        fit(dataset);
    }

    @Override
    public void fit(DataSet dataset)
    {
        Random rand = RandomUtil.getRandom();

        if(ridge < 0)
            throw new IllegalArgumentException("ridge must be positive, not " + ridge);
        final int N = dataset.getSampleSize();
        final int D = dataset.getNumNumericalVars();
        final List<Vec> X = dataset.getDataVectors();

        //Create smaller gram matrix K and decompose is
        basisVecs = sampleBasisVectors(k, dataset, X, method, basisSize, sampleWithReplacment, rand);

        accelCache = k.getAccelerationCache(basisVecs);

        Matrix K = new DenseMatrix(basisSize, basisSize);
        for (int i = 0; i < basisSize; i++)
        {
            K.set(i, i, k.eval(i, i, basisVecs, accelCache));
            for (int j = i + 1; j < basisSize; j++)
            {
                double val = k.eval(i, j, basisVecs, accelCache);
                K.set(i, j, val);
                K.set(j, i, val);
            }
        }

        //Decompose it
        EigenValueDecomposition eig = new EigenValueDecomposition(K);

        double[] eigenVals = eig.getRealEigenvalues();
        DenseVector eigNorm = new DenseVector(eigenVals.length);
        for (int i = 0; i < eigenVals.length; i++)
            eigNorm.set(i, 1.0 / Math.sqrt(Math.max(1e-7, eigenVals[i]+ridge)));

        //U * 1/sqrt(S)
        Matrix U = eig.getV();
        Matrix.diagMult(U, eigNorm);
        transform = U.multiply(eig.getVRaw());
        transform.mutableTranspose();
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected Nystrom(Nystrom toCopy)
    {
        this.k = toCopy.k.clone();
        this.method = toCopy.method;
        this.sampleWithReplacment = toCopy.sampleWithReplacment;
        this.dimension = toCopy.dimension;
        this.ridge = toCopy.ridge;
        this.basisSize = toCopy.basisSize;
        if(toCopy.basisVecs != null)
        {
            this.basisVecs = new ArrayList<Vec>(toCopy.basisVecs.size());
            for(Vec v : toCopy.basisVecs)
                this.basisVecs.add(v.clone());
            if(toCopy.accelCache != null)
                this.accelCache = new DoubleList(toCopy.accelCache);
        }
        if(toCopy.transform != null)
            this.transform = toCopy.transform.clone();
    }
    
    /**
     * Performs sampling of a data set for a subset of the vectors that make a 
     * good set of basis vectors for forming an approximation of a full kernel
     * space. While these methods are motivated from Nystrom's algorithm, they
     * are also useful for others. 
     * 
     * @param k the kernel trick to form the basis for
     * @param dataset the data set to sample from
     * @param X the list of vectors from the data set
     * @param method the sampling method to use
     * @param basisSize the number of basis vectors to select
     * @param sampleWithReplacment whether or not the sample with replacement
     * @param rand the source of randomness for the sampling
     * @return a list of basis vectors sampled from the data set. 
     * @see SamplingMethod
     */
    public static List<Vec> sampleBasisVectors(KernelTrick k, DataSet dataset, final List<Vec> X, SamplingMethod method, int basisSize, boolean sampleWithReplacment, Random rand)
    {
        List<Vec> basisVecs = new ArrayList<Vec>(basisSize);
        final int N = dataset.getSampleSize();
        switch (method)
        {
            case DIAGONAL:
                double[] diags = new double[N];
                diags[0] = k.eval(X.get(0), X.get(0));
                for (int i = 1; i < N; i++)
                    diags[i] = diags[i-1] + k.eval(X.get(i), X.get(i));
                sample(basisSize, rand, diags, X, sampleWithReplacment, basisVecs);
                break;
            case NORM:
                double[] norms = new double[N];
                List<Vec> gramVecs = new ArrayList<Vec>();
                for (int i = 0; i < N; i++)
                    gramVecs.add(new DenseVector(N));

                List<Double> tmpCache = k.getAccelerationCache(X);
                for (int i = 0; i < N; i++)
                {
                    gramVecs.get(i).set(i, k.eval(i, i, X, tmpCache));
                    for (int j = i + 1; j < N; j++)
                    {
                        double val = k.eval(i, j, X, tmpCache);
                        gramVecs.get(i).set(j, val);
                        gramVecs.get(j).set(i, val);
                    }
                }

                norms[0] = gramVecs.get(0).pNorm(2);
                for (int i = 1; i < gramVecs.size(); i++)
                    norms[i] = norms[i - 1] + gramVecs.get(i).pNorm(2);
                sample(basisSize, rand, norms, X, sampleWithReplacment, basisVecs);
                break;
            case KMEANS:
                HamerlyKMeans kMeans = new HamerlyKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.KPP);
                kMeans.setStoreMeans(true);
                kMeans.cluster(dataset, basisSize);
                basisVecs.addAll(kMeans.getMeans());
                break;
            case UNIFORM:
            default:
                if (sampleWithReplacment)
                {
                    Set<Integer> sampled = new IntSet(basisSize);
                    while (sampled.size() < basisSize)
                        sampled.add(rand.nextInt(N));
                    for (int indx : sampled)
                        basisVecs.add(X.get(indx));
                }
                else
                    for (int i = 0; i < basisSize; i++)
                        basisVecs.add(X.get(rand.nextInt(N)));

        }
        return basisVecs;
    }
    
    /**
     * Performs waited sampling on the cumulative sum of all values mapped to 
     * each vector. The sampled vectors will be placed directly into {@link #basisVecs}
     * @param basisSize the number of basis vectors to sample for
     * @param rand the source of randomness
     * @param weightSume the cumulative weight sum
     * @param X the list of vectors
     * @param sampleWithReplacment  whether or no to sample with replacement
     * @param basisVecs the list to store the vecs in
     */
    private static void sample(int basisSize, Random rand, double[] weightSume, List<Vec> X, boolean sampleWithReplacment, List<Vec> basisVecs)
    {
        Set<Integer> sampled = new IntSet(basisSize);
        
        double max = weightSume[weightSume.length-1];
        for(int i = 0; i < basisSize; i++)
        {
            double rndVal = rand.nextDouble()*max;
            int indx = Arrays.binarySearch(weightSume, rndVal);
            if(indx < 0)
                indx = (-(indx) - 1);
            if(sampleWithReplacment)//no need to do anything
                basisVecs.add(X.get(indx));
            else
            {
                int size = sampled.size();
                sampled.add(indx);
                if(sampled.size() == size)
                    i--;//do it again
                else
                    basisVecs.add(X.get(indx));
            }
        }
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec x = dp.getNumericalValues();
        List<Double> qi = k.getQueryInfo(x);
        Vec kVec = new DenseVector(basisVecs.size());
        for(int i = 0; i < basisVecs.size(); i++)
            kVec.set(i, k.eval(i, x, qi, basisVecs, accelCache));
        return new DataPoint(kVec.multiply(transform), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }

    @Override
    public Nystrom clone()
    {
        return new Nystrom(this);
    }

    /**
     * Sets the regularization parameter to add to the eigen values of the gram
     * matrix. This can be particularly useful when using a large (500+) number
     * of components.
     *
     * @param ridge the non-negative value in [0, &infin;) to add to each eigen
     * value
     */
    public void setRidge(double ridge)
    {
        if (ridge < 0 || Double.isNaN(ridge) || Double.isInfinite(ridge))
            throw new IllegalArgumentException("Ridge must be non negative, not " + ridge);
        this.ridge = ridge;
    }

    /**
     * Returns the regularization value added to each eigen value
     *
     * @return the regularization value added to each eigen value
     */
    public double getRidge()
    {
        return ridge;
    }

    /**
     * Sets the dimension of the new feature space, which is the number of
     * principal components to select from the kernelized feature space.
     *
     * @param dimension the number of dimensions to project down too
     */
    public void setDimension(int dimension)
    {
        if (dimension < 1)
            throw new IllegalArgumentException("The number of dimensions must be positive, not " + dimension);
        this.dimension = dimension;
    }

    /**
     * Returns the number of dimensions to project down too
     *
     * @return the number of dimensions to project down too
     */
    public int getDimension()
    {
        return dimension;
    }

    /**
     * Sets the method of selecting the basis vectors
     *
     * @param method the method of selecting the basis vectors
     */
    public void setBasisSamplingMethod(SamplingMethod method)
    {
        this.method = method;
    }

    /**
     * Returns the method of selecting the basis vectors
     *
     * @return the method of selecting the basis vectors
     */
    public SamplingMethod getBasisSamplingMethod()
    {
        return method;
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
}
