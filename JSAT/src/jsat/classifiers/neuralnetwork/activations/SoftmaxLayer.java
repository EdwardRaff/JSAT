package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.MathTricks;

/**
 * This activation layer is meant to be used as the top-most layer for 
 * classification problems, and uses the softmax function (also known as cross
 * entropy) to convert the inputs into probabilities. 
 * 
 * @author Edward Raff
 */
public class SoftmaxLayer implements ActivationLayer
{


	private static final long serialVersionUID = -6595701781466123463L;

	@Override
    public void activate(Vec input, Vec output)
    {
        input.copyTo(output);
        MathTricks.softmax(output, false);
    }

    @Override
    public void backprop(Vec input, Vec output, Vec delta_partial, Vec errout)
    {
        if(delta_partial != errout)//if the same object, nothing to do
            delta_partial.copyTo(errout);
    }

    @Override
    public void activate(Matrix input, Matrix output, boolean rowMajor)
    {
        if(rowMajor)//easy
            for(int i = 0; i < input.rows(); i++)
                activate(input.getRowView(i), output.getRowView(i));
        else//TODO, do this more efficently
            for(int j = 0; j < input.cols(); j++)
                activate(input.getColumnView(j), output.getColumnView(j));
    }

    @Override
    public void backprop(Matrix input, Matrix output, Matrix delta_partial, Matrix errout, boolean rowMajor)
    {
        if(delta_partial != errout)//if the same object, nothing to do
            delta_partial.copyTo(errout);
    }

    @Override
    public SoftmaxLayer clone()
    {
        return new SoftmaxLayer();
    }
    
}
