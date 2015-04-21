
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class TriweightKF implements KernelFunction
{

	private static final long serialVersionUID = -9156392658970318676L;

	private TriweightKF()
    {
    }

    private static class SingletonHolder
    {

        public static final TriweightKF INSTANCE = new TriweightKF();
    }

    /**
     * Returns the singleton instance of this class
     * @return the instance of this class
     */
    public static TriweightKF getInstance()
    {
        return SingletonHolder.INSTANCE;
    }
    
    @Override
    public double k(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return Math.pow(1 - u*u, 3)*(35.0/32.0);
    }

    @Override
    public double intK(double u)
    {
        if(u < -1)
            return 0;
        if(u > 1)
            return 1;
        return (-5*Math.pow(u, 7) + 21*Math.pow(u, 5) - 35 * Math.pow(u, 3) + 35 *u + 16)/32;
    }

    @Override
    public double k2()
    {
        return 1.0/9.0;
    }

    @Override
    public double cutOff()
    {
        return Math.ulp(1)+1;
    }

    @Override
    public double kPrime(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return -u;
    }

    @Override
    public String toString()
    {
        return "Triweight Kernel";
    }
}
