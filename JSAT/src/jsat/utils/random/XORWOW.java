package jsat.utils.random;

import java.util.Random;

/**
 * A very fast PRNG that passes the Diehard tests. It has a period
 * of 2<sup>192</sup>−2<sup>32</sup>
 * <br><br>
 * See: G. Marsaglia. <i>Xorshift RNGs</i>. Journal of Statistical Software, 8, 
 * 14:1–9, 2003
 * @author EdwardRaff
 */
public class XORWOW extends Random
{

    private static final long serialVersionUID = 4516396552618366318L;
    private long x, y, z, w, v, d;

    /**
     * Creates a new PRNG with a random seed
     */
    public XORWOW()
    {
        super();
    }

    /**
     * Creates a new PRNG
     * @param seed the seed that controls the initial state of the PRNG
     * @see #setSeed(long) 
     */
    public XORWOW(long seed)
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
        
        v = super.next(32);
        v = v << 32;
        v += super.next(32);
        
        d = super.next(32);
        d = d << 32;
        d += super.next(32);
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
        t = (x ^ (x >> 2));
        x = y;
        y = z;
        z = w;
        w = v;
        v = (v ^ (v << 4)) ^ (t ^ (t << 1));
        
        t = (d += 362437) + v;
        return t;
    }

    @Override
    public double nextDouble()
    {
        long l = nextLong() >>> 11; 
        return l / (double)(1L << 53);
    }
}
