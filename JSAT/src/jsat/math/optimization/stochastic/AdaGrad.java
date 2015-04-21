package jsat.math.optimization.stochastic;

import jsat.linear.ConstantVector;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * AdaGrad provides an adaptive learning rate for each individual feature<br>
 * <br>
 * See: Duchi, J., Hazan, E.,&amp;Singer, Y. (2011). <i>Adaptive Subgradient 
 * Methods for Online Learning and Stochastic Optimization</i>. Journal of 
 * Machine Learning Research, 12, 2121–2159.
 * 
 * @author Edward Raff
 */
public class AdaGrad implements GradientUpdater
{

	private static final long serialVersionUID = 5138474612999751777L;
	private Vec daigG;
    private double biasG;
    
    /**
     * Creates a new AdaGrad updater
     */
    public AdaGrad()
    {
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public AdaGrad(AdaGrad toCopy)
    {
        if(toCopy.daigG != null)
            this.daigG = toCopy.daigG.clone();
        this.biasG = toCopy.biasG;
    }
    

    @Override
    public void update(Vec x, Vec grad, double eta)
    {
        update(x, grad, eta, 0, 0);
    }

    @Override
    public double update(Vec x, Vec grad, double eta, double bias, double biasGrad)
    {
        for(IndexValue iv : grad)
        {
            final int indx = iv.getIndex();
            final double grad_i = iv.getValue();
            final double g_ii = daigG.get(indx);
            x.increment(indx, -eta*grad_i/Math.sqrt(g_ii));
            daigG.increment(indx, grad_i*grad_i);
        }
        
        double biasUpdate = eta*biasGrad/Math.sqrt(biasG);
        biasG += biasGrad*biasGrad;
        return biasUpdate;
    }

    @Override
    public AdaGrad clone()
    {
        return new AdaGrad(this);
    }

    @Override
    public void setup(int d)
    {
        daigG = new DenseVector(new ConstantVector(1.0, d));
        biasG = 1;
    }
    
}
