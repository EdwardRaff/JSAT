
package jsat.distributions.empirical.kernelfunc;
import static java.lang.Math.*;
import jsat.distributions.Normal;

/**
 *
 * @author Edward Raff
 */
public class GaussKF implements KernelFunction
{

	private static final long serialVersionUID = -6765390012694573184L;

	private GaussKF()
    {
    }

    private static class SingletonHolder
    {

        public static final GaussKF INSTANCE = new GaussKF();
    }

    /**
     * Returns the singleton instance of this class
     * @return the instance of this class
     */
    public static GaussKF getInstance()
    {
        return SingletonHolder.INSTANCE;
    }
    
    @Override
    public double k(double u)
    {
        return Normal.pdf(u, 0, 1);
    }

    @Override
    public double intK(double u)
    {
        return Normal.cdf(u, 0, 1);
    }

    @Override
    public double k2()
    {
        return 1;
    }

    @Override
    public double cutOff()
    {
        /*
         * This is not techincaly correct, as this value of k(u) is still 7.998827757006813E-38
         * However, this is very close to zero, and is so small that  k(u)+x = x, for most values of x. 
         * Unless this probability si going to be near zero, values past this point will have 
         * no effect on the result
         */
        return 13;
    }

    @Override
    public double kPrime(double u)
    {
        return -exp(-pow(u, 2)/2)*u/sqrt(2 * PI);
    }

    @Override
    public String toString()
    {
        return "Gaussian Kernel";
    }
}
