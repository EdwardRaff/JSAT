package jsat.math.optimization.stochastic;

import jsat.linear.DenseVector;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;

/**
 * Performs unaltered Stochastic Gradient Decent updates using either standard 
 * or Nestrov momentum. <br>
 * <br> 
 * See:<br>
 * <ul>
 * <li>Bengio, Y., Boulanger-Lewandowski, N.,&amp;Pascanu, R. (2013). <i>Advances 
 * in optimizing recurrent networks</i>. In 2013 IEEE International Conference 
 * on Acoustics, Speech and Signal Processing (pp. 8624–8628). IEEE. 
 * doi:10.1109/ICASSP.2013.6639349</li>
 * <li>Sutskever, I., Martens, J., Dahl, G.,&amp;Hinton, G. (2013). <i>On the 
 * importance of initialization and momentum in deep learning</i>. JMLR W&amp;CP,
 * 28, 1139–1147.</li>
 * </ul>
 * @author Edward Raff
 */
public class SGDMomentum implements GradientUpdater
{

    private static final long serialVersionUID = -3837883539010356899L;
    
    private double momentum;
    private boolean nestrov;
    private Vec velocity;
    private double biasVelocity;

    /**
     * Creates a new SGD with Momentum learner
     * @param momentum the amount of momentum to use
     * @param nestrov {@code true} to use Nestrov momentum, {@code false} for
     * standard. 
     */
    public SGDMomentum(double momentum, boolean nestrov)
    {
        setMomentum(momentum);
        this.nestrov = nestrov;
    }
    
    /**
     * Creates a new SGD with Nestrov Momentum learner
     * @param momentum the amount of momentum to use
     */
    public SGDMomentum(double momentum)
    {
        this(momentum, true);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SGDMomentum(SGDMomentum toCopy)
    {
        this.momentum = toCopy.momentum;
        if(toCopy.velocity != null)
            this.velocity = toCopy.velocity.clone();
        this.biasVelocity = toCopy.biasVelocity;
    }

    /**
     * Sets the momentum for accumulating gradients. 
     * @param momentum the momentum buildup term in (0, 1)
     */
    public void setMomentum(double momentum)
    {
        if(momentum <= 0 || momentum >= 1 || Double.isNaN(momentum))
            throw new IllegalArgumentException("Momentum must be in (0,1) not " + momentum);
        this.momentum = momentum;
    }

    /**
     * 
     * @return the momentum buildup term
     */
    public double getMomentum()
    {
        return momentum;
    }
    
    @Override
    public void update(Vec x, Vec grad, double eta)
    {
        update(x, grad, eta, 0.0, 0.0);
    }

    @Override
    public double update(Vec x, Vec grad, double eta, double bias, double biasGrad)
    {
        double biasUpdate;
        if (nestrov)
        {
            //update
            x.mutableAdd(momentum * momentum, velocity);
            x.mutableSubtract((1 + momentum) * eta, grad);
            biasUpdate = -momentum*momentum*biasVelocity + (1+momentum)*eta*biasGrad;
        }
        else//clasic momentum
        {
            //update
            x.mutableAdd(momentum, velocity);
            x.mutableSubtract(eta, grad);
            biasUpdate = -momentum*biasVelocity + eta*biasGrad;
        }

        //velocity
        velocity.mutableMultiply(momentum);
        velocity.mutableSubtract(eta, grad);
        biasVelocity = biasVelocity*momentum - eta*biasGrad;
        
        return biasUpdate;
    }

    @Override
    public SGDMomentum clone()
    {
        return new SGDMomentum(this);
    }

    @Override
    public void setup(int d)
    {
        velocity = new ScaledVector(new DenseVector(d));
        biasVelocity = 0;
    }
    
}
