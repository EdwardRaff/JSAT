package jsat.datatransform;

import static jsat.linear.MatrixStatistics.covarianceMatrix;
import static jsat.linear.MatrixStatistics.meanVector;

import java.util.Comparator;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.EigenValueDecomposition;
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.SubMatrix;
import jsat.linear.Vec;

/**
 * An extension of {@link PCA} that attempts to capture the variance, and make
 * the variables in the output space independent from each-other. An of equal
 * scale, so that the covariance is equal to {@link Matrix#eye(int) I}. The
 * results may be further from the identity matrix than desired as the target
 * dimension shrinks<br>
 * <br>
 * The Whitened PCA is more computational expensive than the normal PCA
 * algorithm, but transforming the data takes the same time.
 *
 * @author Edward Raff
 */
public class WhitenedPCA implements DataTransform {

  /**
   * Factory for producing new {@link WhitenedPCA} transforms
   */
  static public class WhitenedPCATransformFactory extends DataTransformFactoryParm {

    private int dimensions;

    /**
     * Creates a new WhitenedPCA Factory
     *
     * @param dims
     *          the number of dimensions to project down to
     */
    public WhitenedPCATransformFactory(final int dims) {
      setDimensions(dims);
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public WhitenedPCATransformFactory(final WhitenedPCATransformFactory toCopy) {
      this(toCopy.dimensions);
    }

    @Override
    public WhitenedPCATransformFactory clone() {
      return new WhitenedPCATransformFactory(this);
    }

    /**
     * Returns the number of dimensions to project down to
     *
     * @return the number of dimensions to project down to
     */
    public int getDimensions() {
      return dimensions;
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      return new WhitenedPCA(dataset, dimensions);
    }

    /**
     * Sets the number of dimensions to project down to
     *
     * @param dimensions
     *          the feature size to project down to
     */
    public void setDimensions(final int dimensions) {
      if (dimensions < 1) {
        throw new IllegalArgumentException("Number of dimensions must be positive, not " + dimensions);
      }
      this.dimensions = dimensions;
    }
  }

  private static final long serialVersionUID = 6134243673037330608L;
  /**
   * Regularization parameter
   */
  protected double regularization;

  /**
   * The number of dimensions to project down to
   */
  protected int dims;

  /**
   * The final transformation matrix, that will create new points <tt>y</tt> =
   * <tt>transform</tt> * x
   */
  protected Matrix transform;

  /**
   * Creates a new WhitenedPCA. The dimensions will be chosen so that the subset
   * of dimensions is of full rank. The regularization parameter will be chosen
   * as the log of the condition of the covariance.
   *
   * @param dataSet
   *          the data set to whiten
   */
  public WhitenedPCA(final DataSet dataSet) {

    final SingularValueDecomposition svd = getSVD(dataSet);
    setRegularization(svd);
    setDims(svd.getRank());

    setUpTransform(svd);
  }

  /**
   * Creates a new WhitenedPCA, the dimensions will be chosen so that the subset
   * of dimensions is of full rank.
   *
   * @param dataSet
   *          the data set to whiten
   * @param regularization
   *          the amount of regularization to add, avoids numerical instability
   */
  public WhitenedPCA(final DataSet dataSet, final double regularization) {
    setRegularization(regularization);
    final SingularValueDecomposition svd = getSVD(dataSet);
    setDims(svd.getRank());

    setUpTransform(svd);
  }

  /**
   * Creates a new WhitenedPCA
   *
   * @param dataSet
   *          the data set to whiten
   * @param regularization
   *          the amount of regularization to add, avoids numerical instability
   * @param dims
   *          the number of dimensions to project down to
   */
  public WhitenedPCA(final DataSet dataSet, final double regularization, final int dims) {
    setRegularization(regularization);
    setDims(dims);

    setUpTransform(getSVD(dataSet));

  }

  /**
   * Creates a new WhitenedPCA. The regularization parameter will be chosen as
   * the log of the condition of the covariance.
   *
   * @param dataSet
   *          the data set to whiten
   * @param dims
   *          the number of dimensions to project down to
   */
  public WhitenedPCA(final DataSet dataSet, final int dims) {

    final SingularValueDecomposition svd = getSVD(dataSet);
    setRegularization(svd);
    setDims(dims);

    setUpTransform(svd);
  }

  /**
   * Copy constructor
   *
   * @param other
   *          the transform to make a copy of
   */
  private WhitenedPCA(final WhitenedPCA other) {
    regularization = other.regularization;
    dims = other.dims;
    transform = other.transform.clone();
  }

  @Override
  public DataTransform clone() {
    return new WhitenedPCA(this);
  }

  /**
   * Gets a SVD for the covariance matrix of the data set
   *
   * @param dataSet
   *          the data set in question
   * @return the SVD for the covariance
   */
  private SingularValueDecomposition getSVD(final DataSet dataSet) {
    final Matrix cov = covarianceMatrix(meanVector(dataSet), dataSet);
    for (int i = 0; i < cov.rows(); i++) {
      for (int j = 0; j < i; j++) {
        cov.set(j, i, cov.get(i, j));
      }
    }
    final EigenValueDecomposition evd = new EigenValueDecomposition(cov);
    // Sort form largest to smallest
    evd.sortByEigenValue(new Comparator<Double>() {
      @Override
      public int compare(final Double o1, final Double o2) {
        return -Double.compare(o1, o2);
      }
    });
    return new SingularValueDecomposition(evd.getVRaw(), evd.getVRaw(), evd.getRealEigenvalues());
  }

  private void setDims(final int dims) {
    if (dims < 1) {
      throw new ArithmeticException("Invalid number of dimensions, bust be > 0");
    }
    this.dims = dims;
  }

  private void setRegularization(final double regularization) {
    if (regularization < 0 || Double.isNaN(regularization) || Double.isInfinite(regularization)) {
      throw new ArithmeticException("Regularization must be non negative value, not " + regularization);
    }
    this.regularization = regularization;
  }

  private void setRegularization(final SingularValueDecomposition svd) {
    if (svd.isFullRank()) {
      setRegularization(1e-10);
    } else {
      setRegularization(Math.max(Math.log(1.0 + svd.getSingularValues()[svd.getRank()]) * 0.25, 1e-4));
    }
  }

  /**
   * Creates the {@link #transform transform matrix} to be used when converting
   * data points. It is called in the constructor after all values are set.
   *
   * @param svd
   *          the SVD of the covariance of the source data set
   */
  protected void setUpTransform(final SingularValueDecomposition svd) {
    final Vec diag = new DenseVector(dims);

    final double[] s = svd.getSingularValues();

    for (int i = 0; i < dims; i++) {
      diag.set(i, 1.0 / Math.sqrt(s[i] + regularization));
    }

    transform = new SubMatrix(svd.getU().transpose(), 0, 0, dims, s.length).clone();

    Matrix.diagMult(diag, transform);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    final Vec newVec = transform.multiply(dp.getNumericalValues());

    final DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());

    return newDp;
  }
}
