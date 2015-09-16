package jsat.classifiers.neuralnetwork.initializers;

import java.util.Random;
import jsat.linear.ConstantVector;
import jsat.linear.Vec;

/**
 * This initializes all bias values to a single constant value
 *
 * @author Edward Raff
 */
public class ConstantInit implements BiastInitializer {

  private static final long serialVersionUID = 2638413936718283757L;
  private double c;

  /**
   *
   * @param c
   *          the constant to set all biases to
   */
  public ConstantInit(final double c) {
    this.c = c;
  }

  @Override
  public ConstantInit clone() {
    return new ConstantInit(c);
  }

  /**
   *
   * @return the constant value that will be used for initialization
   */
  public double getConstant() {
    return c;
  }

  @Override
  public void init(final Vec b, final int fanIn, final Random rand) {
    new ConstantVector(c, b.length()).copyTo(b);
  }

  /**
   *
   * @param c
   *          the constant value to use
   */
  public void setConstant(final double c) {
    if (Double.isNaN(c) || Double.isInfinite(c)) {
      throw new IllegalArgumentException("Constant must be a real value, not " + c);
    }
    this.c = c;
  }

}
