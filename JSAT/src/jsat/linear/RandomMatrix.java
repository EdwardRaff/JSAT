package jsat.linear;

import java.util.Random;
import jsat.utils.random.RandomUtil;

/**
 * Stores a Matrix full of random values in constant O(1) space by re-computing 
 * all matrix values on the fly as need. This allows memory reduction and use 
 * when it is necessary to use the matrix with a large sparse data set, where 
 * some matrix values may never even be used - or used very infrequently. <br>
 * <br>
 * This method is most useful when:
 * <ul>
 * <li>A random matrix can not be fit into main memory</li>
 * <li>An in memory matrix with the model being trained would result in swapping
 * , in which case the slower Random Matrix would be faster since it can avoid 
 * swapping</li>
 * <li>A very large matrix must be synchronized across many threads or machines. 
 * The Random Matrix takes O(1) space and is thread safe</li>
 * <li>Initializing a random dense matrix</li>
 * <li>The accesses of the matrix is sparse enough that not all matrix values
 * will get used, or used very infrequently</li>
 * </ul>
 * <br><br>
 * Because the values of the random matrix are computed on the fly, the Random 
 * Matrix can not be altered. If attempted, an exception will be thrown. 
 * <br><br>
 * Because a Random Matric can not be altered, it can not fulfill the contract 
 * of {@link #getMatrixOfSameType(int, int) }. For this reason, it will return a
 * {@link DenseMatrix} so that use cases of the given method do not break, and 
 * can return new - altered - matrices. 
 * 
 * @author Edward Raff
 */
abstract public class RandomMatrix extends GenericMatrix
{

	private static final long serialVersionUID = 3514801206898749257L;
	/*
     * Implementation note: It is assumed that the default random object is a
     * PRNG with a single word / long of state. A higher quality PRNG cant be 
     * used if it requires too many words of state, as the initalization will 
     * then dominate the computation of every index. 
     */
    private int rows, cols;
    private long seedMult;

    /**
     * Creates a new random matrix object
     * @param rows the number of rows for the random matrix
     * @param cols the number of columns for the random matrix
     */
    public RandomMatrix(int rows, int cols)
    {
        this(rows, cols, RandomUtil.getRandom().nextLong());
    }
    
    /**
     * Creates a new random matrix object
     * @param rows the number of rows for the random matrix
     * @param cols the number of columns for the random matrix
     * @param seedMult a value to multiply with the seed used for each 
     * individual index. It should be a large value
     */
    public RandomMatrix(int rows, int cols, long seedMult)
    {
        if(rows <= 0)
            throw new IllegalArgumentException("rows must be positive, not " + rows);
        if(cols <= 0)
            throw new IllegalArgumentException("cols must be positive, not " + cols);
        this.rows = rows;
        this.cols = cols;
        this.seedMult = seedMult;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RandomMatrix(RandomMatrix toCopy)
    {
        this(toCopy.rows, toCopy.cols, toCopy.seedMult);
    }
    
    private ThreadLocal<Random> localRand = new ThreadLocal<Random>()
    {
        @Override
        protected Random initialValue()
        {
            return new Random(1);//seed will get set by user
        }
    };

    @Override
    protected Matrix getMatrixOfSameType(int rows, int cols)
    {
        return new DenseMatrix(rows, cols);
    }
    
    /**
     * Computes the value of an index given the already initialized 
     * {@link Random} object. This is called by the {@link #get(int, int) } 
     * method, and will make sure that the correct seed is set before calling 
     * this method. 
     * 
     * @param rand the PRNG to generate the index value from
     * @return the value for a given index based on the given PRNG
     */
    abstract protected double getVal(Random rand);


    @Override
    public double get(int i, int j)
    {
        long seed = (i+1)*(j+cols)*seedMult;

        Random rand = localRand.get();
        rand.setSeed(seed);
        return getVal(rand);
    }

    @Override
    public void set(int i, int j, double value)
    {
        throw new UnsupportedOperationException("Random Matrix can not be altered"); 
    }

    @Override
    public int rows()
    {
        return rows;
    }

    @Override
    public int cols()
    {
        return cols;
    }

    @Override
    public boolean isSparce()
    {
        return false;
    }

    @Override
    public boolean canBeMutated()
    {
        return false;
    }

    @Override
    public void changeSize(int newRows, int newCols)
    {
        throw new UnsupportedOperationException("Random Matrix can not be altered"); 
    }
}
