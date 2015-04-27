package jsat.math.optimization.stochastic;

import jsat.linear.Vec;

/**
 * Performs unaltered Stochastic Gradient Decent updates computing 
 * <i>x = x- &eta; grad</i><br>
 * <br>
 * Because the SimpleSGD requires no internal state, it is not necessary to call 
 * {@link #setup(int) }. 
 * 
 * @author Edward Raff
 */
public class SimpleSGD implements GradientUpdater
{


	private static final long serialVersionUID = 4022442467298319553L;

	/**
     * Creates a new SGD updater
     */
    public SimpleSGD()
    {
    }

    @Override
    public void update(Vec x, Vec grad, double eta)
    {
        x.mutableSubtract(eta, grad);
    }

    @Override
    public double update(Vec x, Vec grad, double eta, double bias, double biasGrad)
    {
        x.mutableSubtract(eta, grad);
        return eta*biasGrad;
    }

    @Override
    public SimpleSGD clone()
    {
        return new SimpleSGD();
    }

    @Override
    public void setup(int d)
    {
        //no setup to be done
    }
    
}
