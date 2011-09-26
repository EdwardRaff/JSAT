
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class EpanechnikovKF implements KernelFunction
{

    public double k(double u)
    {
        if(Math.abs(u) > 1)
            return 0;
        return (1-u*u)*(3.0/4.0);
    }

    public double intK(double u)
    {
        if(u < -1)
            return 0;
        if( u > 1)
            return 1;
        return (-u*u*u + 3 *u + 2)/4;
    }

    public double k2()
    {
        return 1.0/5.0;
    }

    public double cutOff()
    {
        return Math.ulp(1)+1;
    }
    
}
