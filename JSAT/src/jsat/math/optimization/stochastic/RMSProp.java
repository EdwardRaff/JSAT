package jsat.math.optimization.stochastic;

import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;

/**
 * rmsprop is an adpative learning weight scheme proposed by Geoffrey Hinton. 
 * Provides an adaptive learning rate for each individual feature
 * 
 * @author Edward Raff
 */
public class RMSProp implements GradientUpdater
{
    private double rho;
    private Vec daigG;
    
    /**
     * Creates a new RMSProp updater that uses a decay rate of 0.9
     */
    public RMSProp()
    {
        this(0.9);
    }

    /**
     * Creates a new RMSProp updater
     * @param rho the decay rate to use
     */
    public RMSProp(double rho)
    {
        setRho(rho);
    }

    /**
     * Sets the decay rate used by rmsprop. Lower values focus more on the 
     * current gradient, where higher values incorporate a longer history. 
     * 
     * @param rho the decay rate in (0, 1) to use
     */
    public void setRho(double rho)
    {
        if(rho <= 0 || rho >= 1 || Double.isNaN(rho))
            throw new IllegalArgumentException("Rho should be a value in (0, 1) not " + rho);
        this.rho = rho;
    }

    /**
     * 
     * @return the decay rate parameter to use
     */
    public double getRho()
    {
        return rho;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RMSProp(RMSProp toCopy)
    {
        if(toCopy.daigG != null)
            this.daigG = toCopy.daigG.clone();
        this.rho = toCopy.rho;
    }
    

    @Override
    public void update(Vec x, Vec grad, double eta)
    {
        daigG.mutableMultiply(rho);
        for(IndexValue iv : grad)
        {
            final int indx = iv.getIndex();
            final double grad_i = iv.getValue();
            daigG.increment(indx, (1-rho)*grad_i*grad_i);
            final double g_ii = daigG.get(indx);
            x.increment(indx, -eta*grad_i/Math.sqrt(g_ii));
        }
    }

    @Override
    public RMSProp clone()
    {
        return new RMSProp(this);
    }

    @Override
    public void setup(int d)
    {
        daigG = new ScaledVector(new DenseVector(d));
    }
    
}
