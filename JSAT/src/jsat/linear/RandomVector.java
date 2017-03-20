package jsat.linear;

import java.util.Random;
import jsat.utils.random.RandomUtil;

/**
 * Stores a Vector full of random values in constant O(1) space by re-computing 
 * all matrix values on the fly as need. This allows memory reduction and use 
 * when it is necessary to use the matrix with a large sparse data set, where 
 * some matrix values may never even be used - or used very infrequently. <br>
 * <br>
 * Because the values of the random vector are computed on the fly, the Random 
 * Vector can not be altered. If attempted, an exception will be thrown. 
 * 
 * @author Edward Raff
 */
public abstract class RandomVector extends Vec
{

	private static final long serialVersionUID = -1587968421978707875L;
	/*
     * Implementation note: It is assumed that the default random object is a
     * PRNG with a single word / long of state. A higher quality PRNG cant be 
     * used if it requires too many words of state, as the initalization will 
     * then dominate the computation of every index. 
     */
    private int length;
    private long seedMult;

    /**
     * Creates a new Random Vector object
     * @param length the length of the vector
     */
    public RandomVector(int length)
    {
        this(length, RandomUtil.getRandom().nextLong());
    }

    /**
     * Creates a new Random Vector object
     * @param length the length of the vector
     * @param seedMult a value to multiply with the seed used for each 
     * individual index. It should be a large value
     */
    public RandomVector(int length, long seedMult)
    {
        if(length<= 0)
            throw new IllegalArgumentException("Vector length must be positive, not " + length);
        this.length = length;
        this.seedMult = seedMult;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected RandomVector(RandomVector toCopy)
    {
        this(toCopy.length, toCopy.seedMult);
    }
    
    private ThreadLocal<Random> localRand = new ThreadLocal<Random>()
    {
        @Override
        protected Random initialValue()
        {
            return new Random(1);//seed will get set by user
        }
    };
    
    /**
     * Computes the value of an index given the already initialized 
     * {@link Random} object. This is called by the {@link #get(int) } 
     * method, and will make sure that the correct seed is set before calling 
     * this method. 
     * 
     * @param rand the PRNG to generate the index value from
     * @return the value for a given index based on the given PRNG
     */
    abstract protected double getVal(Random rand);
    
    @Override
    public double get(int index)
    {
        long seed = (index+length)*seedMult;
        Random rand = localRand.get();
        rand.setSeed(seed);
        return getVal(rand);
    }

    @Override
    public void set(int index, double val)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public int length()
    {
        return length;
    }

    @Override
    public void multiply(double c, Matrix A, Vec b)
    {
        if(this.length() != A.rows())
            throw new ArithmeticException("Vector x Matrix dimensions do not agree [1," + this.length() + "] x [" + A.rows() + ", " + A.cols() + "]");
        if(b.length() != A.cols())
            throw new ArithmeticException("Destination vector is not the right size");
        
        for(int i = 0; i < this.length(); i++)
        {
            double this_i = c*get(i);
            for(int j = 0; j < A.cols(); j++)
                b.increment(j, this_i*A.get(i, j));
        }
    }

    @Override
    public void mutableAdd(double c)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public void mutableAdd(double c, Vec b)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public void mutablePairwiseMultiply(Vec b)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public void mutableMultiply(double c)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public void mutablePairwiseDivide(Vec b)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public void mutableDivide(double c)
    {
        throw new UnsupportedOperationException("RandomVector can not be altered");
    }

    @Override
    public Vec sortedCopy()
    {
        DenseVector dv = new DenseVector(this);
        return dv.sortedCopy();
    }

    @Override
    public double min()
    {
        double min = Double.MAX_VALUE;
        for(IndexValue iv : this)
            min = Math.min(iv.getValue(), min);
        return min;
    }

    @Override
    public double max()
    {
        double max = -Double.MAX_VALUE;
        for(IndexValue iv : this)
            max = Math.min(iv.getValue(), max);
        return max;
    }

    @Override
    public boolean isSparse()
    {
        return false;
    }

    @Override
    abstract public Vec clone();

    @Override
    public double dot(Vec v)
    {
        double dot = 0;

        for (IndexValue iv : v)
            dot += get(iv.getIndex()) * iv.getValue();
        return dot;
    }

    @Override
    public boolean canBeMutated()
    {
        return false;
    }
}
