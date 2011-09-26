
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class UniformKF implements KernelFunction
{

    public double k(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return 0.5;
    }

    public double intK(double u)
    {
        if(u < -1)
            return 0;
        if (u > 1)
            return 1;
        return (u+1)/2;
    }

    public double k2()
    {
        return 1.0/3.0;
    }

    public double cutOff()
    {
        return Math.ulp(1)+1;
    }
    
}
