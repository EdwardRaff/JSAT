
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class UniformKF implements KernelFunction
{

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
    
}
