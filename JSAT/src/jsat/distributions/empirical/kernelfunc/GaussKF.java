
package jsat.distributions.empirical.kernelfunc;
import jsat.distributions.Normal;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class GaussKF implements KernelFunction
{

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
    
}
