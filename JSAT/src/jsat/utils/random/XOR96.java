package jsat.utils.random;

import java.util.Random;

/**
 * A fast PRNG that produces medium quality random numbers. It has a period of
 * 2<sup>96</sup>-1
 * <br><br>
 * See: G. Marsaglia. <i>Xorshift RNGs</i>. Journal of Statistical Software, 8, 
 * 14:1–9, 2003
 * @author Edward Raff
 */
public class XOR96 extends Random
{

    private static final long serialVersionUID = 1247900882148980639L;

    private static final long a = 13, b = 19, c = 3;//magic from paper

    private long x, y, z;

    /**
     * Creates a new PRNG with a random seed
     */
    public XOR96()
    {
        super();
    }

    /**
     * Creates a new PRNG
     * @param seed the seed that controls the initial state of the PRNG
     * @see #setSeed(long) 
     */
    public XOR96(long seed)
    {
        super(seed);
    }

    @Override
    public synchronized void setSeed(long seed)
    {
        super.setSeed(seed);
        x = super.next(32);
        x = x << 32;
        x += super.next(32);
        
        y = super.next(32);
        y = y << 32;
        y += super.next(32);
        
        z = super.next(32);
        z = z << 32;
        z += super.next(32);
    }
    
    @Override
    protected int next(int bits)
    {
        return (int)(nextLong() >>> (64 - bits));
    }

    @Override
    public long nextLong()
    {
        long t = (x ^ (x << a));
        x = y;
        y = z;
        z = (z ^ (z >>> c)) ^ (t ^ (t >>> b));
        return z;
    }
    
    @Override
    public double nextDouble()
    {
        long l = nextLong() >>> 11; 
        return l / (double)(1L << 53);
    }
}
