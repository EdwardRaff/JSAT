package jsat;

import java.util.Random;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.utils.random.XORWOW;

/**
 * Contains pre determined code for generating specific data sets. The form and
 * values of the data set are fixed, and do not need to be specified. Training
 * and testing sets are generated by the same methods.
 *
 * @author Edward Raff
 */
public class FixedProblems {

  private static final Vec c2l_m0 = new DenseVector(new double[] { 12, 14, 25, 31, 10, 9, 1 });
  private static final Vec c2l_m1 = new DenseVector(new double[] { -9, -7, -13, -6, -11, -9, -1 });
  private static final NormalM c2l_c0 = new NormalM(c2l_m0, Matrix.eye(c2l_m0.length()));
  private static final NormalM c2l_c1 = new NormalM(c2l_m1, Matrix.eye(c2l_m0.length()));

  /**
   * Generates a linearly separable binary classification problem
   *
   * @param dataSetSize
   *          the number of points to generated
   * @param rand
   *          the source of randomness
   * @return a binary classification data set that is linearly separable
   */
  public static ClassificationDataSet get2ClassLinear(final int dataSetSize, final Random rand) {
    final ClassificationDataSet train = new ClassificationDataSet(c2l_m0.length(), new CategoricalData[0],
        new CategoricalData(2));

    for (final Vec s : c2l_c0.sample(dataSetSize, rand)) {
      train.addDataPoint(s, new int[0], 0);
    }
    for (final Vec s : c2l_c1.sample(dataSetSize, rand)) {
      train.addDataPoint(s, new int[0], 1);
    }

    return train;
  }

  public static ClassificationDataSet getCircles(final int dataSetSize, final double... radi) {
    return getCircles(dataSetSize, new XORWOW(), radi);
  }

  public static ClassificationDataSet getCircles(final int dataSetSize, final Random rand, final double... radi) {
    final ClassificationDataSet train = new ClassificationDataSet(2, new CategoricalData[0],
        new CategoricalData(radi.length));

    final int n = dataSetSize / 2;

    for (int r_i = 0; r_i < radi.length; r_i++) {
      for (int i = 0; i < n; i++) {
        final double t = 2 * Math.PI * i / n;
        final double x = radi[r_i] * Math.cos(t) + (rand.nextDouble() - 0.5) / 5;
        final double y = radi[r_i] * Math.sin(t) + (rand.nextDouble() - 0.5) / 5;
        train.addDataPoint(DenseVector.toDenseVec(x, y), new int[0], r_i);
      }
    }

    return train;
  }

  public static ClassificationDataSet getHalfCircles(final int dataSetSize, final Random rand, final double... radi) {
    final ClassificationDataSet train = new ClassificationDataSet(2, new CategoricalData[0],
        new CategoricalData(radi.length));

    final int n = dataSetSize / 2;

    for (int r_i = 0; r_i < radi.length; r_i++) {
      for (int i = 0; i < n; i++) {
        final double t = 2 * Math.PI * (i / 2) / n;
        final double x = radi[r_i] * Math.cos(t) + (rand.nextDouble() - 0.5) / 5;
        final double y = radi[r_i] * Math.sin(t) + (rand.nextDouble() - 0.5) / 5;
        train.addDataPoint(DenseVector.toDenseVec(x, y), new int[0], r_i);
      }
    }

    return train;
  }

  /**
   * Returns a classification problem with small uniform noise where there is a
   * small circle of radius 1 within a circle of radius 4. Each circle shares
   * the same center.
   *
   * @param dataSetSize
   *          the even number of data points to create
   * @param rand
   *          the source of randomness
   * @return a classification data set with two classes
   */
  public static ClassificationDataSet getInnerOuterCircle(final int dataSetSize, final Random rand) {
    return getInnerOuterCircle(dataSetSize, rand, 1, 4);
  }

  public static ClassificationDataSet getInnerOuterCircle(final int dataSetSize, final Random rand, final double r1,
      final double r2) {
    return getCircles(dataSetSize, rand, r1, r2);
  }

  /**
   * Generates a regression problem that can be solved by linear regression
   * methods
   *
   * @param dataSetSize
   *          the number of data points to get
   * @param rand
   *          the randomness to use
   * @return a regression data set
   */
  public static RegressionDataSet getLinearRegression(final int dataSetSize, final Random rand) {
    return getLinearRegression(dataSetSize, rand, c2l_m0);
  }

  /**
   * Generates a regression problem that can be solved by linear regression
   * methods
   *
   * @param dataSetSize
   *          the number of data points to get
   * @param rand
   *          the randomness to use
   * @param coef
   *          the coefficients to use for the linear regression
   * @return a regression data set
   */
  public static RegressionDataSet getLinearRegression(final int dataSetSize, final Random rand, final Vec coef) {
    final RegressionDataSet rds = new RegressionDataSet(coef.length(), new CategoricalData[0]);

    for (int i = 0; i < dataSetSize; i++) {
      final Vec s = new DenseVector(coef.length());
      for (int j = 0; j < s.length(); j++) {
        s.set(j, rand.nextDouble());
      }
      rds.addDataPoint(s, new int[0], s.dot(coef));
    }

    return rds;
  }

  /**
   * Generates a linearly separable multi class problem
   *
   * @param dataSetSize
   *          the number of data points to generate per class
   * @param K
   *          the number of classes to generate
   * @return a new multi class data set
   */
  public static ClassificationDataSet getSimpleKClassLinear(final int dataSetSize, final int K) {
    return getSimpleKClassLinear(dataSetSize, K, new XORWOW());
  }

  /**
   * Generates a linearly separable multi class problem
   *
   * @param dataSetSize
   *          the number of data points to generate per class
   * @param K
   *          the number of classes to generate
   * @param rand
   *          the source of randomness
   * @return a new multi class data set
   */
  public static ClassificationDataSet getSimpleKClassLinear(final int dataSetSize, final int K, final Random rand) {
    final ClassificationDataSet train = new ClassificationDataSet(K, new CategoricalData[0], new CategoricalData(K));
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < dataSetSize; i++) {
        final Vec dv = Vec.random(K, rand);
        dv.set(k, 10 + rand.nextGaussian());
        train.addDataPoint(dv, new int[0], k);
      }
    }
    return train;
  }

  /**
   * Creates a new Regression problem of the for x<sub>2</sub>+c sin(x
   * <sub>2</sub>) = y. It is meant to be an easy test case for non-linear
   * regression algorithms.
   *
   * @param dataSetSize
   *          the number of data points to generate
   * @param rand
   *          the source of randomness
   * @return a new regression data set
   */
  public static RegressionDataSet getSimpleRegression1(final int dataSetSize, final Random rand) {
    final RegressionDataSet rds = new RegressionDataSet(2, new CategoricalData[0]);
    for (int i = 0; i < dataSetSize; i++) {
      final Vec s = new DenseVector(new double[] { rand.nextDouble() * 4, rand.nextDouble() * 4 });
      rds.addDataPoint(s, new int[0], s.get(0) + 4 * Math.cos(s.get(1)));
    }
    return rds;
  }
}
