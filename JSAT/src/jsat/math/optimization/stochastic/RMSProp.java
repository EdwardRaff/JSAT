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

	private static final long serialVersionUID = 3512851084092042727L;
	private double rho;
    private Vec daigG;
    private double biasG;
    
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
    public RMSProp(final double rho)
    {
        setRho(rho);
    }

    /**
     * Sets the decay rate used by rmsprop. Lower values focus more on the 
     * current gradient, where higher values incorporate a longer history. 
     * 
     * @param rho the decay rate in (0, 1) to use
     */
    public void setRho(final double rho)
    {
        if(rho <= 0 || rho >= 1 || Double.isNaN(rho)) {
          throw new IllegalArgumentException("Rho should be a value in (0, 1) not " + rho);
        }
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
    public RMSProp(final RMSProp toCopy)
    {
        if(toCopy.daigG != null) {
          this.daigG = toCopy.daigG.clone();
        }
        this.rho = toCopy.rho;
        this.biasG = toCopy.biasG;
    }
    

    @Override
    public void update(final Vec x, final Vec grad, final double eta)
    {
        update(x, grad, eta, 0, 0);
    }

    @Override
    public double update(final Vec x, final Vec grad, final double eta, final double bias, final double biasGrad)
    {
        daigG.mutableMultiply(rho);
        for(final IndexValue iv : grad)
        {
            final int indx = iv.getIndex();
            final double grad_i = iv.getValue();
            daigG.increment(indx, (1-rho)*grad_i*grad_i);
            final double g_iiRoot = Math.max(Math.sqrt(daigG.get(indx)), Math.abs(grad_i));//tiny grad sqrd could result in zero
            x.increment(indx, -eta*grad_i/g_iiRoot);
        }
        
        biasG *= rho;
        biasG += (1-rho)*biasGrad*biasGrad;
        final double g_iiRoot = Math.max(Math.sqrt(biasG), Math.abs(biasGrad));//tiny grad sqrd could result in zero
        return eta*biasGrad/g_iiRoot;
    }

    @Override
    public RMSProp clone()
    {
        return new RMSProp(this);
    }

    @Override
    public void setup(final int d)
    {
        daigG = new ScaledVector(new DenseVector(d));
        biasG = 0;
    }
    
}
