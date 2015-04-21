package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This Activation Layer is for <b>Re</b>ctified <b>L</b>inear <b>U</b>nits. A
 * ReLU activation is simply f(x) = max(0, x), and is thus very fast to compute.
 * <br>
 * See: Nair, V., &amp; Hinton, G. E. (2010). <i>Rectified Linear Units Improve 
 * Restricted Boltzmann Machines</i>. Proceedings of the 27th International 
 * Conference on Machine Learning, 807–814.
 * @author Edward Raff
 */
public class ReLU implements ActivationLayer
{


	private static final long serialVersionUID = -6691240473485759789L;

	@Override
    public void activate(Vec input, Vec output)
    {
        for(int i = 0; i < input.length(); i++)
            output.set(i, Math.max(0, input.get(i)));
    }
    
    @Override
    public void activate(Matrix input, Matrix output, boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++)
            for(int j = 0; j < input.cols(); j++)
                output.set(i, j, Math.max(0, input.get(i, j)));
    }

    @Override
    public void backprop(Vec input, Vec output, Vec delta_partial, Vec errout)
    {
        for(int i = 0; i < input.length(); i++)
        {
            double out_i = output.get(i);
            if(out_i != 0)
                errout.set(i, delta_partial.get(i));
            else
                errout.set(i, 0.0);
        }
    }

    @Override
    public void backprop(Matrix input, Matrix output, Matrix delta_partial, Matrix errout, boolean rowMajor)
    {
        for (int i = 0; i < input.rows(); i++)
            for (int j = 0; j < input.cols(); j++)
                if (output.get(i, j) != 0)
                    errout.set(i, j, delta_partial.get(i, j));
                else
                    errout.set(i, j, 0.0);
    }

    @Override
    public ReLU clone()
    {
        return new ReLU();
    }
    
}
