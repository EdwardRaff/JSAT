package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;

/**
 * An extension of {@link WhitenedPCA}, is the Whitened Zero Component Analysis.
 * Whitened ZCA can not project to a lower dimension, as it rotates the output
 * in the original dimension.
 *
 * @author Edward Raff
 */
public class WhitenedZCA extends WhitenedPCA implements InPlaceTransform {

  /**
   * Factory for producing new {@link WhitenedZCA} transforms.
   */
  static public class WhitenedZCATransformFactory extends DataTransformFactoryParm {

    private double reg;
    private boolean autoReg;

    /**
     * Creates a new WhitenedZCA where the regularization will be determined
     * automatically
     */
    public WhitenedZCATransformFactory() {
      this(1.0);
      autoReg = false;
    }

    /**
     * Creates a new WhitenedZCA factory that will use the regularization value
     * provided
     *
     * @param reg
     *          the regularization to use
     */
    public WhitenedZCATransformFactory(final double reg) {
      setRegularization(reg);
      autoReg = true;
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public WhitenedZCATransformFactory(final WhitenedZCATransformFactory toCopy) {
      reg = toCopy.reg;
      autoReg = toCopy.autoReg;
    }

    @Override
    public WhitenedZCATransformFactory clone() {
      return new WhitenedZCATransformFactory(this);
    }

    /**
     * Returns the amount of regularization that will be used if
     * {@link #isAutoReg() } is {@code false}.
     *
     * @return the amount of regularization that will be used
     */
    public double getRegularization() {
      return reg;
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      if (autoReg) {
        return new WhitenedZCA(dataset);
      }
      return new WhitenedZCA(dataset, reg);
    }

    /**
     * Returns whether or not the regularization parameter is determined
     * automatically.
     *
     * @return {@code true} if the regularization is determined automatically
     */
    public boolean isAutoReg() {
      return autoReg;
    }

    /**
     * Sets whether or not to automatically select a regularization value.
     *
     * @param autoReg
     *          {@code true} to automatically select a regularization vale,
     *          {@code false} to use the value set by
     *          {@link #setRegularization(double) }
     */
    public void setAutoReg(final boolean autoReg) {
      this.autoReg = autoReg;
    }

    /**
     * Sets the amount of regularization to use, this value will be ignored if
     * {@link #setAutoReg(boolean) } is set to {@code true}
     *
     * @param reg
     *          the positive regularization parameter
     */
    public void setRegularization(final double reg) {
      if (reg <= 0 || Double.isNaN(reg) || Double.isInfinite(reg)) {
        throw new IllegalArgumentException("Regularization must be a positive value, not " + reg);
      }
      this.reg = reg;
    }
  }

  private static final long serialVersionUID = 7546033727733619587L;

  private final ThreadLocal<Vec> tempVecs;

  /**
   * Creates a new Whitened ZCA. The regularization parameter will be chosen as
   * the log of the condition of the covariance.
   *
   * @param dataSet
   *          the data set to whiten
   */
  public WhitenedZCA(final DataSet dataSet) {
    super(dataSet);
    tempVecs = getThreadLocal(dataSet.getNumNumericalVars());
  }

  /**
   * Creates a new Whitened ZCA.
   *
   * @param dataSet
   *          the data set to whiten
   * @param regularization
   *          the amount of regularization to add, avoids numerical instability
   */
  public WhitenedZCA(final DataSet dataSet, final double regularization) {
    super(dataSet, regularization);
    tempVecs = getThreadLocal(dataSet.getNumNumericalVars());
  }

  private ThreadLocal<Vec> getThreadLocal(final int dim) {
    return new ThreadLocal<Vec>() {

      @Override
      protected Vec initialValue() {
        return new DenseVector(dim);
      }
    };
  }

  @Override
  public void mutableTransform(final DataPoint dp) {
    final Vec target = tempVecs.get();
    target.zeroOut();
    transform.multiply(dp.getNumericalValues(), 1.0, target);
    target.copyTo(dp.getNumericalValues());
  }

  @Override
  public boolean mutatesNominal() {
    return false;
  }

  @Override
  protected void setUpTransform(final SingularValueDecomposition svd) {
    final double[] s = svd.getSingularValues();
    final Vec diag = new DenseVector(s.length);

    for (int i = 0; i < s.length; i++) {
      diag.set(i, 1.0 / Math.sqrt(s[i] + regularization));
    }

    final Matrix U = svd.getU();

    transform = U.multiply(Matrix.diag(diag)).multiply(U.transpose());
  }
}
