package jsat.utils.random;

import java.util.Random;

/**
 * A Complement-Multiply-With-Carry PRNG. It is a fast high quality generator 
 * that passes the Diehard tests. It has a period of over 2<sup>131083</sup>
 * <br><br>
 * See: Marsaglia, G. (2005). 
 * <i>On the randomness of Pi and other decimal expansions</i>. Interstat 5
 * 
 * @author Edward Raff
 */
public class CMWC4096 extends Random
{

    private static final long serialVersionUID = -5061963074440046713L;
    private static final long a = 18782;
    private int c = 362436;
    private int i = 4095;
    private int[] Q;

    /**
     * Creates a new PRNG with a random seed
     */
    public CMWC4096()
    {
        super();
    }
    
    /**
     * Creates a new PRNG
     * @param seed the seed that controls the initial state of the PRNG
     * @see #setSeed(long) 
     */
    public CMWC4096(long seed)
    {
        super(seed);
    }
    

    @Override
    public synchronized void setSeed(long seed)
    {
        super.setSeed(seed);
        if(Q == null)
            Q = new int[4096];
        for (int j = 0; j < Q.length; j++)
            Q[j] = super.next(32);
    }
    
    @Override
    protected int next(int bits)
    {
        long t;

        long x, r = 0xfffffffe;
        i = (i + 1) & 4095;
        t = a * Q[i] + c;
        c = (int) (t >>> 32);
        x = t + c;
        if (x < c)
        {
            x++;
            c++;
        }
        return (Q[i] = (int) (r - x)) >>> 32 - bits;
    }    

}
