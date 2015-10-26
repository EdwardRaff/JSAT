package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This layer provides the standard Sigmoid activation f(x) =
 * 1/(1+exp(-x))
 * 
 * @author Edward Raff
 */
public class SigmoidLayer implements ActivationLayer
{


	private static final long serialVersionUID = 160273287445169627L;

	@Override
    public void activate(final Vec input, final Vec output)
    {
        for(int i = 0; i < input.length(); i++) {
          output.set(i, 1/(1+Math.exp(-input.get(i))));
        }
    }
    
    @Override
    public void activate(final Matrix input, final Matrix output, final boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++) {
          for (int j = 0; j < input.cols(); j++) {
            output.set(i, j, 1.0/(1+Math.exp(-input.get(i, j))));
          }
        }
    }

    @Override
    public void backprop(final Vec input, final Vec output, final Vec delta_partial, final Vec errout)
    {
        for(int i = 0; i < input.length(); i++)
        {
            final double out_i = output.get(i);
            final double errin_i = delta_partial.get(i);
            errout.set(i, out_i*(1-out_i)*errin_i);
        }
    }

    

    @Override
    public void backprop(final Matrix input, final Matrix output, final Matrix delta_partial, final Matrix errout, final boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++) {
          for(int j = 0; j < input.cols(); j++)
          {
            final double out_ij = output.get(i, j);
            final double errin_ij = delta_partial.get(i, j);
            errout.set(i, j, out_ij*(1-out_ij)*errin_ij);
          }
        }
    }

    @Override
    public SigmoidLayer clone()
    {
        return new SigmoidLayer();
    }
    
}
