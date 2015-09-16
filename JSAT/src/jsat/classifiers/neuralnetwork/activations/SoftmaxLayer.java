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
public class SoftmaxLayer implements ActivationLayer {

  private static final long serialVersionUID = -6595701781466123463L;

  @Override
  public void activate(final Matrix input, final Matrix output, final boolean rowMajor) {
    if (rowMajor) {// easy
      for (int i = 0; i < input.rows(); i++) {
        activate(input.getRowView(i), output.getRowView(i));
      }
    } else {
      for (int j = 0; j < input.cols(); j++) {
        activate(input.getColumnView(j), output.getColumnView(j));
      }
    }
  }

  @Override
  public void activate(final Vec input, final Vec output) {
    input.copyTo(output);
    MathTricks.softmax(output, false);
  }

  @Override
  public void backprop(final Matrix input, final Matrix output, final Matrix delta_partial, final Matrix errout,
      final boolean rowMajor) {
    if (delta_partial != errout) {// if the same object, nothing to do
      delta_partial.copyTo(errout);
    }
  }

  @Override
  public void backprop(final Vec input, final Vec output, final Vec delta_partial, final Vec errout) {
    if (delta_partial != errout) {// if the same object, nothing to do
      delta_partial.copyTo(errout);
    }
  }

  @Override
  public SoftmaxLayer clone() {
    return new SoftmaxLayer();
  }

}
