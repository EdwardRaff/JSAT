package jsat.utils.random;

import java.util.Random;

/**
 * A fast PRNG that produces medium quality random numbers that passes the 
 * diehard tests. It has a period of 2<sup>128</sup>-1
 * <br><br>
 * See: G. Marsaglia. <i>Xorshift RNGs</i>. Journal of Statistical Software, 8, 
 * 14:1–9, 2003
 * @author Edward Raff
 */
public class XOR128 extends Random
{

    private static final long serialVersionUID = -5218902638864900490L;
    private long x, y, z, w;

    /**
     * Creates a new PRNG with a random seed
     */
    public XOR128()
    {
        super();
    }

    /**
     * Creates a new PRNG
     * @param seed the seed that controls the initial state of the PRNG
     * @see #setSeed(long) 
     */
    public XOR128(long seed)
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
        
        w = super.next(32);
        w = w << 32;
        w += super.next(32);
    }

    @Override
    protected int next(int bits)
    {
        return (int)(nextLong() >>> (64 - bits));
    }

    @Override
    public long nextLong()
    {
        long t;
        t = (x ^ (x << 11));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >>> 19)) ^ (t ^ (t >>> 8));
        return w;
    }
    
    @Override
    public double nextDouble()
    {
        long l = nextLong() >>> 11; 
        return l / (double)(1L << 53);
    }
}
