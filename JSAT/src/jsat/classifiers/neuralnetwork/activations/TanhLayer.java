package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This layer provides the standard tanh activation f(x) =
 * tanh(x)
 * 
 * @author Edward Raff
 */
public class TanhLayer implements ActivationLayer
{


	private static final long serialVersionUID = -8369008344962638121L;

	@Override
    public void activate(final Vec input, final Vec output)
    {
        for(int i = 0; i < input.length(); i++) {
          output.set(i, Math.tanh(input.get(i)));
        }
    }

    @Override
    public void activate(final Matrix input, final Matrix output, final boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++) {
          for (int j = 0; j < input.cols(); j++) {
            output.set(i, j, Math.tanh(input.get(i, j)));
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
            errout.set(i, (1-out_i*out_i)*errin_i);
        }
    }
    
    @Override
    public void backprop(final Matrix input, final Matrix output, final Matrix delta_partial, final Matrix errout, final boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++) {
          for (int j = 0; j < input.cols(); j++)
          {
            final double out_ij = output.get(i, j);
            final double errin_ij = delta_partial.get(i, j);
            errout.set(i, j, (1-out_ij*out_ij)*errin_ij);
          }
        }
    }

    @Override
    public TanhLayer clone()
    {
        return new TanhLayer();
    }
    
}
