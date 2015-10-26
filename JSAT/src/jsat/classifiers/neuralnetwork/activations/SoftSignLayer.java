package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This provides the Soft Sign activation function f(x) = x/(1+abs(x)), which is 
 * similar to the {@link TanhLayer tanh} activation and has a min/max of -1 and
 * 1. However it is significantly faster to compute. <br>
 * <br>
 * See: Glorot, X., &amp; Bengio, Y. (2010). <i>Understanding the difficulty of 
 * training deep feedforward neural networks</i>. Journal of Machine Learning 
 * Research - Proceedings Track, 9, 249–256. Retrieved from 
 * <a href="http://jmlr.csail.mit.edu/proceedings/papers/v9/glorot10a/glorot10a.pdf">
 * here</a>
 * @author Edward Raff
 */
public class SoftSignLayer implements ActivationLayer
{


	private static final long serialVersionUID = 9137125423044227288L;

	@Override
    public void activate(final Vec input, final Vec output)
    {
        for(int i = 0; i < input.length(); i++)
        {
            final double in_i = input.get(i);
            output.set(i, in_i/(1.0+Math.abs(in_i)));
        }
    }

    @Override
    public void activate(final Matrix input, final Matrix output, final boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++) {
          for(int j = 0; j < input.cols(); j++)
          {
            final double in_ij = input.get(i, j);
            output.set(i, j, in_ij/(1.0+Math.abs(in_ij)));
          }
        }
    }
    
    @Override
    public void backprop(final Vec input, final Vec output, final Vec delta_partial, final Vec errout)
    {
        for(int i = 0; i < input.length(); i++)
        {
            final double tmp_i = (1-Math.abs(output.get(i)));
            final double errin_i = delta_partial.get(i);
            errout.set(i, tmp_i*tmp_i*errin_i);
        }
    }

    @Override
    public void backprop(final Matrix input, final Matrix output, final Matrix delta_partial, final Matrix errout, final boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++) {
          for (int j = 0; j < input.cols(); j++)
          {
            final double tmp_ij = (1 - Math.abs(output.get(i, j)));
            final double errin_ij = delta_partial.get(i, j);
            errout.set(i, j, tmp_ij*tmp_ij*errin_ij);
          }
        }
    }

    @Override
    public SoftSignLayer clone()
    {
        return new SoftSignLayer();
    }
    
}
