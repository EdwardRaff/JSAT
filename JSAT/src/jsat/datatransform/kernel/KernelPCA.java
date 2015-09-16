package jsat.datatransform.kernel;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.DataTransformFactoryParm;
import jsat.datatransform.PCA;
import jsat.datatransform.kernel.Nystrom.SamplingMethod;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.EigenValueDecomposition;
import jsat.linear.Matrix;
import jsat.linear.RowColumnOps;
import jsat.linear.Vec;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.utils.random.XOR96;

/**
 * A kernelized implementation of {@link PCA}. Because this works in a different
 * feature space, it will do its own centering in the kernel space. <br>
 * <br>
 * KernelPCA is expensive to compute at O(n<sup>3</sup>) work, where <i>n</i> is
 * the number of data points. For this reason, sampling from {@link Nystrom} is
 * used to reduce the data set to a reasonable approximation. <br>
 * <br>
 * See: Schölkopf, B., Smola, A.,&amp;Müller, K.-R. (1998). <i>Nonlinear
 * Component Analysis as a Kernel Eigenvalue Problem</i>. Neural Computation,
 * 10(5), 1299–1319. doi:10.1162/089976698300017467
 *
 * @author Edward Raff
 * @see Nystrom.SamplingMethod
 */
public class KernelPCA implements DataTransform {

  /**
   * Factory for producing new {@link KernelPCA} transforms
   */
  static public class KernelPCATransformFactory extends DataTransformFactoryParm {

    @ParameterHolder
    private final KernelTrick k;
    private int dimension;
    private int basisSize;
    private Nystrom.SamplingMethod method;

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public KernelPCATransformFactory(final KernelPCATransformFactory toCopy) {
      this(toCopy.k.clone(), toCopy.dimension, toCopy.basisSize, toCopy.method);
    }

    /**
     * Creates a new Kernel PCA factory
     *
     * @param k
     *          the kernel trick to use
     * @param dimension
     *          the number of dimension to project down to
     * @param basisSize
     *          the number of points from the data set to select. If larger than
     *          the number of data points in the data set, the whole data set
     *          will be used.
     * @param samplingMethod
     *          the sampling method to select the basis vectors
     */
    public KernelPCATransformFactory(final KernelTrick k, final int dimension, final int basisSize,
        final Nystrom.SamplingMethod samplingMethod) {
      this.k = k;
      setDimension(dimension);
      setBasisSize(basisSize);
      setBasisSamplingMethod(samplingMethod);
    }

    @Override
    public KernelPCATransformFactory clone() {
      return new KernelPCATransformFactory(this);
    }

    /**
     * Returns the method of selecting the basis vectors
     *
     * @return the method of selecting the basis vectors
     */
    public SamplingMethod getBasisSamplingMethod() {
      return method;
    }

    /**
     * Returns the number of basis vectors to use
     *
     * @return the number of basis vectors to use
     */
    public int getBasisSize() {
      return basisSize;
    }

    /**
     * Returns the number of dimensions to project down too
     *
     * @return the number of dimensions to project down too
     */
    public int getDimension() {
      return dimension;
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      return new KernelPCA(k, dataset, dimension, basisSize, method);
    }

    /**
     * Sets the method of selecting the basis vectors
     *
     * @param method
     *          the method of selecting the basis vectors
     */
    public void setBasisSamplingMethod(final SamplingMethod method) {
      this.method = method;
    }

    /**
     * Sets the basis size for the Kernel PCA to be learned from. Increasing the
     * basis increase the accuracy of the transform, but increased the training
     * time at a cubic rate.
     *
     * @param basisSize
     *          the number of basis vectors to build Kernel PCA from
     */
    public void setBasisSize(final int basisSize) {
      if (basisSize < 1) {
        throw new IllegalArgumentException("The basis size must be positive, not " + basisSize);
      }
      this.basisSize = basisSize;
    }

    /**
     * Sets the dimension of the new feature space, which is the number of
     * principal components to select from the kernelized feature space.
     *
     * @param dimension
     *          the number of dimensions to project down too
     */
    public void setDimension(final int dimension) {
      if (dimension < 1) {
        throw new IllegalArgumentException("The number of dimensions must be positive, not " + dimension);
      }
      this.dimension = dimension;
    }
  }

  private static final long serialVersionUID = 5676602024560381043L;

  /**
   * The dimension to project down to
   */
  private int dimensions;
  private KernelTrick k;
  private double[] eigenVals;
  /**
   * The matrix of transformed eigen vectors
   */
  private Matrix eigenVecs;

  /**
   * The vecs used for the transform
   */
  private Vec[] vecs;
  // row / colum info for centering in the feature space
  private double[] rowAvg;

  private double allAvg;

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  protected KernelPCA(final KernelPCA toCopy) {
    dimensions = toCopy.dimensions;
    k = toCopy.k.clone();
    eigenVals = Arrays.copyOf(toCopy.eigenVals, toCopy.eigenVals.length);
    eigenVecs = toCopy.eigenVecs.clone();
    vecs = new Vec[toCopy.vecs.length];
    for (int i = 0; i < vecs.length; i++) {
      vecs[i] = toCopy.vecs[i].clone();
    }
    rowAvg = Arrays.copyOf(toCopy.rowAvg, toCopy.rowAvg.length);
    allAvg = toCopy.allAvg;
  }

  /**
   * Creates a new Kernel PCA transform object
   *
   * @param k
   *          the kernel trick to use
   * @param ds
   *          the data set to form the data transform from
   * @param dimensions
   *          the number of dimensions to project down to. Must be less than
   *          than the basis size
   * @param basisSize
   *          the number of points from the data set to select. If larger than
   *          the number of data points in the data set, the whole data set will
   *          be used.
   * @param samplingMethod
   *          the sampling method to select the basis vectors
   */
  public KernelPCA(final KernelTrick k, final DataSet ds, final int dimensions, final int basisSize,
      final Nystrom.SamplingMethod samplingMethod) {
    this.dimensions = dimensions;
    this.k = k;

    if (ds.getSampleSize() <= basisSize) {
      vecs = new Vec[ds.getSampleSize()];
      for (int i = 0; i < vecs.length; i++) {
        vecs[i] = ds.getDataPoint(i).getNumericalValues();
      }
    } else {
      int i = 0;
      final List<Vec> sample = Nystrom.sampleBasisVectors(k, ds, ds.getDataVectors(), samplingMethod, basisSize, false,
          new XOR96());
      vecs = new Vec[sample.size()];
      for (final Vec v : sample) {
        vecs[i++] = v;
      }
    }
    final Matrix K = new DenseMatrix(vecs.length, vecs.length);

    // Info used to compute centered Kernel matrix
    rowAvg = new double[K.rows()];
    allAvg = 0;

    for (int i = 0; i < K.rows(); i++) {
      final Vec x_i = vecs[i];
      for (int j = i; j < K.cols(); j++) {
        final double K_ij = k.eval(x_i, vecs[j]);
        K.set(i, j, K_ij);

        K.set(j, i, K_ij);// K = K'
      }
    }

    // Get row / col info to perform centering. Since K is symetric, the row
    // and col info are the same
    for (int i = 0; i < K.rows(); i++) {
      for (int j = 0; j < K.cols(); j++) {
        rowAvg[i] += K.get(i, j);
      }
    }

    for (int i = 0; i < K.rows(); i++) {
      allAvg += rowAvg[i];
      rowAvg[i] /= K.rows();
    }

    allAvg /= K.rows() * K.cols();

    // Centered version of the marix
    // K_c(i, j) = K_ij - sum_z K_zj / m - sum_z K_iz / m + sum_{z,y} K_zy / m^2
    for (int i = 0; i < K.rows(); i++) {
      for (int j = 0; j < K.cols(); j++) {
        K.set(i, j, K.get(i, j) - rowAvg[i] - rowAvg[j] + allAvg);
      }
    }

    final EigenValueDecomposition evd = new EigenValueDecomposition(K);
    evd.sortByEigenValue(new Comparator<Double>() {
      @Override
      public int compare(final Double o1, final Double o2) {
        return -Double.compare(o1, o2);
      }
    });

    eigenVals = evd.getRealEigenvalues();
    eigenVecs = evd.getV();
    for (int j = 0; j < eigenVals.length; j++) {
      // TODO row order would be more cache friendly
      RowColumnOps.divCol(eigenVecs, j, Math.sqrt(eigenVals[j]));
    }
  }

  @Override
  public KernelPCA clone() {
    return new KernelPCA(this);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    final Vec oldVec = dp.getNumericalValues();
    final Vec newVec = new DenseVector(dimensions);

    // TODO put this in a thread local object? Or hope JVM puts a large array on
    // the stack?
    final double[] kEvals = new double[vecs.length];

    double tAvg = 0;

    for (int j = 0; j < vecs.length; j++) {
      tAvg += kEvals[j] = k.eval(vecs[j], oldVec);
    }

    tAvg /= vecs.length;

    for (int i = 0; i < dimensions; i++) {
      double val = 0;
      for (int j = 0; j < vecs.length; j++) {
        val += eigenVecs.get(j, i) * (kEvals[j] - tAvg - rowAvg[i] + allAvg);
      }
      newVec.set(i, val);
    }

    return new DataPoint(newVec, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
  }
}
