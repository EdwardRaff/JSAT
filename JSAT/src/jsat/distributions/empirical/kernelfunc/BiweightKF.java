
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class BiweightKF implements KernelFunction
{

    public double k(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return Math.pow(1-u*u, 2)*(15.0/16.0);
    }

    public double intK(double u)
    {
        if(u <  -1)
            return 0;
        if(u > 1)
            return 1;
        return Math.pow(u+1, 3)/16.0 * (3*u*u - 9*u + 8);
    }

    public double k2()
    {
        return 1.0/7.0;
    }

    public double cutOff()
    {
        return Math.ulp(1)+1;
    }
    
}
