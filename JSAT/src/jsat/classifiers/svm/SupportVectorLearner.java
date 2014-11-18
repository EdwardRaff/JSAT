
package jsat.classifiers.svm;

import java.util.*;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.Vec;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.utils.ListUtils;

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
public abstract class SupportVectorLearner 
{
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
    private LinkedHashMap<Integer, double[]> partialCache;
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
            setCacheValue((int) Math.min(Math.max(1, bytes/bytesPerRow), Integer.MAX_VALUE));
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
            accelCache = null;
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
                fullCache[i] = new double[N-i];
            
            for(int i = 0; i < N; i++)
                for(int j = i; j < N; j++)
                    fullCache[i][j-i] = k(i, j);
        }
        else if(cacheMode == CacheMode.ROWS && vecs != null)
        {
            partialCache = new LinkedHashMap<Integer, double[]>(N, 0.75f, true)
            {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, double[]> eldest)
                {
                    boolean removeEldest = size() > cacheConst;
                    if(removeEldest)
                    {
                        availableRow = eldest.getValue();
                        for(int i = 0; i < availableRow.length; i++)
                            if(!Double.isNaN(availableRow[i]))
                                availableRow[i] = Double.NaN;
                        cacheEvictions++;
                    }
                    return removeEldest;
                }
            };
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
            
            return fullCache[a][b-a];
        }
        else if(cacheMode == CacheMode.ROWS)
        {
            double[] cache = partialCache.get(a);
            if (cache == null)//try seeing if b has a row present
            {
                double[] b_cache = partialCache.get(b);
                if (b_cache != null)
                    if (Double.isNaN(b_cache[a]))
                        return b_cache[a] = k(a, b);
                    else
                        return b_cache[a];
            }
            //else, neither are in - lets go with a

            if (cache == null)//not present
            {
                //get a row
                if (availableRow != null)
                {
                    cache = availableRow;
                    availableRow = null;
                }
                else
                {
                    cache = new double[vecs.size()];
                    Arrays.fill(cache, Double.NaN);
                }
                
                partialCache.put(a, cache);
                
                if (Double.isNaN(cache[a]))
                    return cache[a] = k(a, b);
                else
                    return cache[a];
                
            }
        }
        return k(a, b);
    }
    
    /**
     * Internal kernel eval source
     * @param a the first vector index
     * @param b the second vector index
     * @return the kernel evaluation of k(a, b)
     */
    private double k(int a, int b)
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
