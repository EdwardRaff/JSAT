
package jsat.classifiers.svm;

import java.io.Serializable;
import java.util.*;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.LinearKernel;
import jsat.linear.Vec;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.utils.DoubleList;
import jsat.utils.ListUtils;
import jsat.utils.concurrent.ConcurrentCacheLRU;

/**
 * Base class for support vector style learners. This means that the learner
 * performs batch training on a fixed set of training points using a
 * {@link KernelTrick kernel} to project the data into a different space. The
 * final set of vectors used may or may not be sparse. It does not necessarily
 * have to be a Support Vector machine.
 * <br><br>
 * This class provides caching mechanism to transparently provide faster kernel.
 *
 * @author Edward Raff
 */
public abstract class SupportVectorLearner implements Serializable
{
    static final long serialVersionUID = 210140232301130063L;

    //Implementation note, NaN is used to indicate a cache value that has not been computed yet.
    @ParameterHolder
    private KernelTrick kernel;
    /**
     * The array of vectors. In the training phase, this should be the set of
     * all training vectors. After training, this should contain only the set of
     * support vectors.
     */
    protected List<Vec> vecs;
    /**
     * The array of coefficients associated with each support vector. This
     * should be instantiated directly when training. When the set of alphas and
     * support vectors is finalized, {@link #setAlphas(double[]) } should be
     * called with a reference to itself or the array where the final alphas are
     * stored. This will initialized any accelerating structures so that
     * {@link #kEvalSum(jsat.linear.Vec) } can be called.
     */
    protected double[] alphas;
    private CacheMode cacheMode;

    /**
     * Kernel evaluation acceleration cache
     */
    protected List<Double> accelCache = null;

    private double[][] fullCache;
    /**
     * Stores rows of a cache matrix.
     */
    private ConcurrentCacheLRU<Integer, double[]> partialCache;
    /**
     * We allow algorithms that know they are going to access a specific row to
     * hint, and save that row in this object to avoid overhead of hitting the
     * LRU. See {@link #accessingRow(int) }
     */
    private double[] specific_row_cache_values = null;
    /**
     * The row that has been explicitly cached
     */
    private int specific_row_cache_row = -1;
    
    /**
     * Holds an available row for inserting into the cache, null if not
     * available. All values already set to Nan
     */
    private double[] availableRow;
    private int cacheConst = 500;

    /**
     * Sets the final set of alphas, and indicates that the final accelerating
     * structures (if available) should be constructed for performing kernel
     * evaluations against unseen vectors.
     * <br>
     * This may be called multiple times in an online scenario, but calls will
     * involve a re-construction of the whole cache.
     *
     * @param alphas the final array of alphas
     */
    protected void setAlphas(double[] alphas)
    {
        this.alphas = alphas;
        accelCache = kernel.getAccelerationCache(vecs);
    }

    /**
     * Determines how the final kernel values are cached. The total number of
     * raw kernel evaluations can be tracked using {@link #evalCount}<br>
     * {@link #setCacheMode(jsat.classifiers.svm.SupportVectorLearner.CacheMode) }
     * should be called before training begins by the training algorithm as
     * described in the method documentation.
     */
    public enum CacheMode
    {
        /**
         * No kernel value caching will be performed.
         */
        NONE,
        /**
         * The entire kernel matrix will be created and cached ahead of time.
         * This is the best option if your data set is small and the kernel
         * cache can fit into memory.
         */
        FULL,
        /**
         * Only the most recently used rows of the kernel matrix will be cached
         * (LRU). When a call to {@link #k(int, int) } occurs, the first value
         * will be taken to be the row of the matrix. <br>
         * Because the kernel matrix is symmetric, if a cache miss occurs - the
         * column value will be checked for its existence. If the row is
         * present, it will be used instead. If both rows are not present, then
         * a new row is inserted for the first index, and another row evicted if
         * necessary.
         * <br>
         * The {@link #cacheEvictions} indicates how many times a row has been
         * evicted from the cache.
         * <br>
         * Row values are computed lazily.
         */
        ROWS
    };

    /**
     * This constructor is meant manly for Serialization to work. It uses a
     * linear kernel and no caching.
     */
    protected SupportVectorLearner()
    {
        this(new LinearKernel(), CacheMode.NONE);
    }

    /**
     * Creates a new Support Vector Learner
     * @param kernel the kernel trick to use
     * @param cacheMode the kernel caching method to use
     */
    public SupportVectorLearner(KernelTrick kernel, CacheMode cacheMode)
    {
        this.cacheMode = cacheMode;
        setKernel(kernel);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SupportVectorLearner(SupportVectorLearner toCopy)
    {
        if(toCopy.kernel != null)
            this.kernel = toCopy.kernel.clone();
        if(toCopy.vecs != null)
        {
            this.vecs = new ArrayList<Vec>(toCopy.vecs.size());
            for(Vec v : toCopy.vecs)
                this.vecs.add(v.clone());
        }
        if(toCopy.alphas != null)
            this.alphas = Arrays.copyOf(toCopy.alphas, toCopy.alphas.length);
        this.cacheMode = toCopy.cacheMode;
        if(toCopy.accelCache != null)
            this.accelCache = new DoubleList(toCopy.accelCache);
        if(toCopy.fullCache != null)
        {
            this.fullCache = new double[toCopy.fullCache.length][];
            for(int i = 0; i < toCopy.fullCache.length; i++)
                this.fullCache[i] = Arrays.copyOf(toCopy.fullCache[i], toCopy.fullCache[i].length);
        }
        if(toCopy.partialCache != null)//TODO handling this better needs to be done
        {
            setCacheMode(cacheMode);
            //        if(toCopy.availableRow != null)
//            this.availableRow = Arrays.copyOf(toCopy.availableRow, toCopy.availableRow.length);
        }

        this.cacheConst = toCopy.cacheConst;


    }

    /**
     * Sets the kernel trick to use
     * @param kernel the kernel trick to use
     */
    public void setKernel(KernelTrick kernel)
    {
        this.kernel = kernel;
    }

    /**
     * Sets the cache value, which may be interpreted differently by different
     * caching schemes. <br>
     * This is currently only used for {@link CacheMode#ROWS}, where the value
     * indicates how many rows will be cached.
     *
     * @param cacheValue the cache value to be used
     */
    public void setCacheValue(int cacheValue)
    {
        this.cacheConst = cacheValue;
    }

    /**
     * Sets the {@link #setCacheValue(int) cache value} to one that will use the
     * specified amount of memory. If the amount of memory specified is great
     * enough, this method will automatically set the
     * {@link #setCacheMode(jsat.classifiers.svm.SupportVectorLearner.CacheMode)
     * cache mode} to {@link CacheMode#FULL}.
     *
     * @param N the number of data points
     * @param bytes the number of bytes of memory to make the cache
     */
    public void setCacheSize(long N, long bytes)
    {
        int DS = Double.SIZE/8;
        bytes /= DS;//Gets the total number of doubles we can store
        if(bytes > N*N/2)
            setCacheMode(CacheMode.FULL);
        else//How many rows can we handle?
        {
            //guessing 2 work overhead for object header + one pointer reference to the array, asusming 64 bit
            long bytesPerRow = N*DS+3*Long.SIZE/8;
            int rows = (int) Math.min(Math.max(1, bytes/bytesPerRow), Integer.MAX_VALUE);
            if(rows > 25)
                setCacheValue(rows);
            else//why bother? just use NONE 
                setCacheMode(CacheMode.NONE);
        }
    }

    /**
     * Returns the current cache value
     * @return the current cache value
     */
    public int getCacheValue()
    {
        return cacheConst;
    }


    /**
     * Returns the current caching mode in use
     * @return the current caching mode in use
     */
    public CacheMode getCacheMode()
    {
        return cacheMode;
    }

    /**
     * Calling this sets the method of caching that will be used. <br>
     * This is called called by the implementing class to initialize and clear
     * the caches. Calling this with the current cache mode will initialize the
     * caches. Once training is complete, call again with {@code null} to
     * deinitialize the caches.
     *
     * @param cacheMode
     */
    public void setCacheMode(CacheMode cacheMode)
    {
        if(cacheMode == null)
        {
            fullCache = null;
            partialCache = null;
            availableRow = null;
            return;
        }
        this.cacheMode = cacheMode;

        if(vecs != null)
            accelCache = kernel.getAccelerationCache(vecs);
        evalCount = 0;
        cacheEvictions = 0;

        final int N = vecs == null ? 0 : vecs.size();

        if(cacheMode == CacheMode.FULL && vecs != null)
        {
            fullCache = new double[N][];
            for(int i = 0; i < N; i++)
            {
                fullCache[i] = new double[N-i];
                Arrays.fill(fullCache[i], Double.NaN);
            }

            //Switched to lazy init, hence NaN above
//            for(int i = 0; i < N; i++)
//                for(int j = i; j < N; j++)
//                    fullCache[i][j-i] = k(i, j);
        }
        else if(cacheMode == CacheMode.ROWS && vecs != null)
        {
            partialCache = new ConcurrentCacheLRU<Integer, double[]>(cacheConst);
        }
        else if(cacheMode == CacheMode.NONE)
            fullCache = null;
    }

    protected int evalCount = 0;
    protected int cacheEvictions = 0;

    public KernelTrick getKernel()
    {
        return kernel;
    }

    /**
     * Performs a summation of the form <br>
     * <big>&#8721;</big> &alpha;<sub>i</sub> k(x<sub>i</sub>, y) <br>
     * for each support vector and associated alpha value currently stored in
     * the support vector machine. It is not necessary to call
     * {@link #setAlphas(double[]) } before calling this, but kernel evaluations
     * may be slower if this is not done.
     * @param y the vector to perform the kernel product sum against
     * @return the sum of the scaled kernel products
     */
    protected double kEvalSum(Vec y)
    {
        if (alphas == null)
            throw new RuntimeException("alphas have not been set");

        return kernel.evalSum(vecs, accelCache, alphas, y, 0, alphas.length);
    }

    /**
     * Performs a kernel evaluation of the product between two vectors directly.
     * This is the slowest way to do a kernel evaluation, and should be avoided
     * unless there is a specific reason to do so.
     * <br>
     * These evaluations will not be counted in {@link #evalCount}
     * @param a the first vector
     * @param b the second vector
     * @return the kernel evaluation of k(a, b)
     */
    protected double kEval(Vec a, Vec b)
    {
        return kernel.eval(a, b);
    }

    /**
     * Performs a kernel evaluation of the a'th and b'th vectors in the
     * {@link #vecs} array.
     *
     * @param a the first vector index
     * @param b the second vector index
     * @return the kernel evaluation of k(a, b)
     */
    protected double kEval(int a, int b)
    {
        if(cacheMode == CacheMode.FULL)
        {
            if(a > b)
            {
                int tmp = a;
                a = b;
                b = tmp;
            }

            double val = fullCache[a][b-a];
            if(Double.isNaN(val))//lazy init
                return fullCache[a][b-a] = k(a, b);
            return val;
        }
        else if(cacheMode == CacheMode.ROWS)
        {
            double[] cache;
            if(specific_row_cache_row == a)
                cache = specific_row_cache_values;
            else
                cache = partialCache.get(a);
            if (cache == null)//not present
            {
                //make a row
                cache = new double[vecs.size()];
                Arrays.fill(cache, Double.NaN);

                double[] cache_missed = partialCache.putIfAbsentAndGet(a, cache);
                if(cache_missed != null)
                    cache = cache_missed;

                if (Double.isNaN(cache[b]))
                    return cache[b] = k(a, b);
                else
                    return cache[b];

            }
        }
        return k(a, b);
    }
    
    /**
     * This method allows the caller to hint that they are about to access many
     * kernel values for a specific row. The row may be selected out from the
     * cache into its own location to avoid excess LRU overhead. Giving a
     * negative index indicates that we are done with the row, and removes it.
     * This method may be called multiple times with different row values. But
     * when done accessing a specific row, a negative value should be passed in.
     *
     *
     * @param r the row to cache explicitly to avoid LRU overhead. Or a negative
     * value to indicate that we are done with any specific row.
     */
    protected void accessingRow(int r)
    {
        if (r < 0)
        {
            specific_row_cache_row = -1;
            specific_row_cache_values = null;
            return;
        }
        
        if(cacheMode == CacheMode.ROWS)
        {
            double[] cache = partialCache.get(r);
            if (cache == null)//not present
            {
                //make a row
                cache = new double[vecs.size()];
                Arrays.fill(cache, Double.NaN);

                double[] cache_missed = partialCache.putIfAbsentAndGet(r, cache);
                if(cache_missed != null)
                    cache = cache_missed;
            }
            specific_row_cache_values = cache;
            specific_row_cache_row = r;
        }
    }

    /**
     * Internal kernel eval source. Only call directly if you KNOW you will not
     * be re-using the resulting value and intentionally wish to skip the
     * caching system
     *
     * @param a the first vector index
     * @param b the second vector index
     * @return the kernel evaluation of k(a, b)
     */
    protected double k(int a, int b)
    {
        evalCount++;
        return kernel.eval(a, b, vecs, accelCache);
    }

    /**
     * Sparsifies the SVM by removing the vectors with &alpha; = 0 from the
     * dataset.
     */
    protected void sparsify()
    {
        final int N = vecs.size();
        int accSize = accelCache == null ? 0 : accelCache.size()/N;
        int svCount = 0;
        for(int i = 0; i < N; i++)
            if(alphas[i] != 0)//Its a support vector
            {
                ListUtils.swap(vecs, svCount, i);
                if(accelCache != null)
                    for(int j = i*accSize; j < (i+1)*accSize; j++)
                        ListUtils.swap(accelCache, svCount*accSize+j-i*accSize, j);
                alphas[svCount++] = alphas[i];
            }

        vecs = new ArrayList<Vec>(vecs.subList(0, svCount));
        alphas = Arrays.copyOfRange(alphas, 0, svCount);
    }
}
