
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class UniformKF implements KernelFunction
{

	private static final long serialVersionUID = -6413579643511350896L;

	private UniformKF()
    {
    }

    private static class SingletonHolder
    {

        public static final UniformKF INSTANCE = new UniformKF();
    }

    /**
     * Returns the singleton instance of this class
     * @return the instance of this class
     */
    public static UniformKF getInstance()
    {
        return SingletonHolder.INSTANCE;
    }

    @Override
    public double k(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return 0.5;
    }

    @Override
    public double intK(double u)
    {
        if(u < -1)
            return 0;
        if (u > 1)
            return 1;
        return (u+1)/2;
    }

    @Override
    public double k2()
    {
        return 1.0/3.0;
    }

    @Override
    public double cutOff()
    {
        return Math.ulp(1)+1;
    }

    @Override
    public double kPrime(double u)
    {
        return 0;
    }

    @Override
    public String toString()
    {
        return "Uniform Kernel";
    }
}
