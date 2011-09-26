
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class TriweightKF implements KernelFunction
{

    public double k(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return Math.pow(1 - u*u, 3)*(35.0/32.0);
    }

    public double intK(double u)
    {
        if(u < -1)
            return 0;
        if(u > 1)
            return 1;
        return (-5*Math.pow(u, 7) + 21*Math.pow(u, 5) - 35 * Math.pow(u, 3) + 35 *u + 16)/32;
    }

    public double k2()
    {
        return 1.0/9.0;
    }

    public double cutOff()
    {
        return Math.ulp(1)+1;
    }
    
}
