package jsat.math.optimization.stochastic;

import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;

/**
 * AdaDelta is inspired by {@link AdaGrad} and was developed for use primarily 
 * in neural networks. It still maintains a per feature learning rate, however 
 * unlike AdaGrad the learning rates may increase over time and are highly 
 * robust to any individual learning rate. <br>
 * <br>
 * See: Zeiler, M. D. (2012). <i>ADADELTA: An Adaptive Learning Rate Method</i>.
 * CoRR, abs/1212.5.
 * 
 * @author Edward Raff
 */
public class AdaDelta implements GradientUpdater
{

	private static final long serialVersionUID = 5855631993426837618L;
	private double rho;
    private Vec gSqrd;
    private Vec deltaXSqrt;
    private double biasGSqrd;
    private double deltaBiasSqrt;
    private double eps = 0.0001;
    

    /**
     * Creates a new AdaDelta updater using a decay rate of 0.95
     */
    public AdaDelta()
    {
        this(0.95);
    }

    /**
     * Creates a new AdaDelta updater
     * @param rho the decay rate to use
     */
    public AdaDelta(double rho)
    {
        setRho(rho);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public AdaDelta(AdaDelta toCopy)
    {
        this.rho = toCopy.rho;
        if(toCopy.gSqrd != null)
        {
            this.gSqrd = toCopy.gSqrd.clone();
            this.deltaXSqrt = toCopy.deltaXSqrt.clone();
        }
        this.biasGSqrd = toCopy.biasGSqrd;
        this.deltaBiasSqrt = toCopy.deltaBiasSqrt;
    }
    
    /**
     * Sets the decay rate used by AdaDelta. Lower values focus more on the 
     * current gradient, where higher values incorporate a longer history. 
     * 
     * @param rho the decay rate in (0, 1) to use
     */
    public void setRho(double rho)
    {
        if(rho <= 0 || rho >= 1 || Double.isNaN(rho))
            throw new IllegalArgumentException("Rho must be in (0, 1)");
        this.rho = rho;
    }

    /**
     * 
     * @return the decay rate that will be used
     */
    public double getRho()
    {
        return rho;
    }

    @Override
    public void update(Vec x, Vec grad, double eta)
    {
        update(x, grad, eta, 0, 0);
    }

    @Override
    public double update(Vec x, Vec grad, double eta, double bias, double biasGrad)
    {
        gSqrd.mutableMultiply(rho);
        biasGSqrd *= rho;
        for(IndexValue iv : grad)
        {
            final int indx = iv.getIndex();
            final double grad_i = iv.getValue();
            gSqrd.increment(indx, grad_i*grad_i*(1-rho));//step 4
            final double gSqrd_i = gSqrd.get(indx);
            final double deltaX_i = deltaXSqrt.get(indx);
            
            final double newDeltaX_i = -Math.sqrt((deltaX_i+eps)/(gSqrd_i+eps))*grad_i;//step 5
            x.increment(indx, eta*newDeltaX_i);//step 7
            deltaXSqrt.increment(indx, (1-rho)/rho*newDeltaX_i*newDeltaX_i);//step 6, using (1-rho)/rho so we can multiply by rho at the end to get the correct result  
        }
        //step 6 correction, apply rho to the left hand side
        deltaXSqrt.mutableMultiply(rho);
        
        //bias term
        biasGSqrd += biasGrad*biasGrad*(1-rho);
        double newDeltaBias = Math.sqrt((deltaBiasSqrt+eps)/(biasGSqrd+eps))*biasGrad;
        double biasUpdate = eta*newDeltaBias;
        deltaBiasSqrt += (1-rho)/rho*newDeltaBias*newDeltaBias;
        deltaBiasSqrt *= rho;
        
        return biasUpdate;
    }

    @Override
    public AdaDelta clone()
    {
        return new AdaDelta(this);
    }

    @Override
    public void setup(int d)
    {
        gSqrd = new ScaledVector(new DenseVector(d));
        deltaXSqrt = new ScaledVector(new DenseVector(d));
        deltaBiasSqrt = biasGSqrd = 0;
    }
    
}
