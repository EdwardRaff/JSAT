package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;

/**
 * The PUK kernel is an alternative to the RBF Kernel. By altering the 
 * {@link #setOmega(double) omega} parameter the behavior of the PUK kernel can 
 * be controlled. The {@link #setSigma(double) sigma} parameter works in the 
 * same way as the RBF Kernel.<br>
 * <br>
 * See: Üstün, B., Melssen, W. J., & Buydens, L. M. C. (2006). <i>Facilitating 
 * the application of Support Vector Regression by using a universal Pearson VII
 * function based kernel</i>. Chemometrics and Intelligent Laboratory Systems, 
 * 81(1), 29–40. doi:10.1016/j.chemolab.2005.09.003
 * 
 * @author Edward Raff
 */
public class PukKernel extends BaseL2Kernel
{
    private double sigma;
    private double omega;
    private double cnst;

    /**
     * Creates a new PUK Kernel
     * @param sigma the width parameter of the kernel
     * @param omega the shape parameter of the kernel
     */
    public PukKernel(double sigma, double omega)
    {
        setSigma(sigma);
        setOmega(omega);
    }

    /**
     * Sets the omega parameter value, which controls the shape of the kernel
     * @param omega the positive parameter value
     */
    public void setOmega(double omega)
    {
        if(omega <= 0 || Double.isNaN(omega) || Double.isInfinite(omega))
            throw new ArithmeticException("omega must be positive, not " + omega);
        this.omega = omega;
        this.cnst = Math.sqrt(Math.pow(2, 1/omega)-1);
    }

    public double getOmega()
    {
        return omega;
    }

    /**
     * Sets the sigma parameter value, which controls the width of the kernel
     * @param sigma the positive parameter value
     */
    public void setSigma(double sigma)
    {
        if(sigma <= 0 || Double.isNaN(sigma) || Double.isInfinite(sigma))
            throw new ArithmeticException("sigma must be positive, not " + sigma);
        this.sigma = sigma;
    }

    public double getSigma()
    {
        return sigma;
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        return getVal(a.pNormDist(2.0, b));
    }

    @Override
    public PukKernel clone()
    {
        return new PukKernel(sigma, omega);
    }

    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return getVal(Math.sqrt(getSqrdNorm(a, b, qi, vecs, cache)));
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        return getVal(Math.sqrt(getSqrdNorm(b, b, trainingSet, cache)));
    }

    private double getVal(double pNormDist)
    {
        double tmp = 2*pNormDist*cnst/sigma;
        return 1/Math.pow(1+tmp*tmp, omega);
    }
    
}
