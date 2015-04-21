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
    public void activate(Vec input, Vec output)
    {
        for(int i = 0; i < input.length(); i++)
        {
            double in_i = input.get(i);
            output.set(i, in_i/(1.0+Math.abs(in_i)));
        }
    }

    @Override
    public void activate(Matrix input, Matrix output, boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++)
            for(int j = 0; j < input.cols(); j++)
            {
                double in_ij = input.get(i, j);
                output.set(i, j, in_ij/(1.0+Math.abs(in_ij)));
            }
    }
    
    @Override
    public void backprop(Vec input, Vec output, Vec delta_partial, Vec errout)
    {
        for(int i = 0; i < input.length(); i++)
        {
            double tmp_i = (1-Math.abs(output.get(i)));
            double errin_i = delta_partial.get(i);
            errout.set(i, tmp_i*tmp_i*errin_i);
        }
    }

    @Override
    public void backprop(Matrix input, Matrix output, Matrix delta_partial, Matrix errout, boolean rowMajor)
    {
        for(int i = 0; i < input.rows(); i++)
            for (int j = 0; j < input.cols(); j++)
            {
                double tmp_ij = (1 - Math.abs(output.get(i, j)));
                double errin_ij = delta_partial.get(i, j);
                errout.set(i, j, tmp_ij*tmp_ij*errin_ij);
            }
    }

    @Override
    public SoftSignLayer clone()
    {
        return new SoftSignLayer();
    }
    
}
