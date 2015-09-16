package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;

/**
 * PNormNormalization transformation performs normalizations of a vector x by
 * one its p-norms where p is in (0, Infinity)
 *
 * @author Edward Raff
 */
public class PNormNormalization implements InPlaceTransform {

  /**
   * Factor for producing {@link PNormNormalization} transforms
   */
  static public class PNormNormalizationFactory extends DataTransformFactoryParm {

    private double p;

    /**
     * Creates a new p norm factory
     *
     * @param p
     *          the norm to use
     */
    public PNormNormalizationFactory(final double p) {
      this.p = p;
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public PNormNormalizationFactory(final PNormNormalizationFactory toCopy) {
      p = toCopy.p;
    }

    @Override
    public PNormNormalizationFactory clone() {
      return new PNormNormalizationFactory(this);
    }

    /**
     * Returns the p-norm that the vectors will be normalized by
     *
     * @return the p-norm that the vectors will be normalized by
     */
    public double getPNorm() {
      return p;
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      return new PNormNormalization(p);
    }

    /**
     * Sets the norm that the vector should be normalized by.
     *
     * @param p
     *          the norm to use in (0, Infinity]
     */
    public void setPNorm(final double p) {
      if (p <= 0 || Double.isNaN(p)) {
        throw new IllegalArgumentException("p must be greater than zero, not " + p);
      }
      this.p = p;
    }

  }

  private static final long serialVersionUID = 2934569881395909607L;

  private final double p;

  /**
   * Creates a new p norm
   *
   * @param p
   *          the norm to use
   */
  public PNormNormalization(final double p) {
    if (p <= 0 || Double.isNaN(p)) {
      throw new IllegalArgumentException("p must be greater than zero, not " + p);
    }
    this.p = p;
  }

  @Override
  public PNormNormalization clone() {
    return new PNormNormalization(p);
  }

  @Override
  public void mutableTransform(final DataPoint dp) {
    final Vec vec = dp.getNumericalValues();
    final double norm = vec.pNorm(p);
    if (norm != 0) {
      vec.mutableDivide(norm);
    }
  }

  @Override
  public boolean mutatesNominal() {
    return false;
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    final DataPoint dpNew = dp.clone();

    mutableTransform(dpNew);
    return dpNew;
  }
}
