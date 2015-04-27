package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class LinearLayer implements ActivationLayer
{


	private static final long serialVersionUID = -4040058095010471379L;

	@Override
    public void activate(Vec input, Vec output)
    {
        input.copyTo(output);
    }

    @Override
    public void activate(Matrix input, Matrix output, boolean rowMajor)
    {
        input.copyTo(output);
    }
    
    @Override
    public void backprop(Vec input, Vec output, Vec delta_partial, Vec errout)
    {
        delta_partial.copyTo(errout);
    }

    @Override
    public void backprop(Matrix input, Matrix output, Matrix delta_partial, Matrix errout, boolean rowMajor)
    {
        delta_partial.copyTo(errout);
    }

    @Override
    public LinearLayer clone()
    {
        return new LinearLayer();
    }
    
}
