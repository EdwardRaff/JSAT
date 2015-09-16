package jsat.classifiers.neuralnetwork.activations;

import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class LinearLayer implements ActivationLayer {

  private static final long serialVersionUID = -4040058095010471379L;

  @Override
  public void activate(final Matrix input, final Matrix output, final boolean rowMajor) {
    input.copyTo(output);
  }

  @Override
  public void activate(final Vec input, final Vec output) {
    input.copyTo(output);
  }

  @Override
  public void backprop(final Matrix input, final Matrix output, final Matrix delta_partial, final Matrix errout,
      final boolean rowMajor) {
    delta_partial.copyTo(errout);
  }

  @Override
  public void backprop(final Vec input, final Vec output, final Vec delta_partial, final Vec errout) {
    delta_partial.copyTo(errout);
  }

  @Override
  public LinearLayer clone() {
    return new LinearLayer();
  }

}
