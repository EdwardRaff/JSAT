package jsat.distributions.multivariate;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.DataPoint;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.math.OnLineStatistics;
import jsat.parameters.DoubleParameter;
import jsat.parameters.IntParameter;
import jsat.parameters.KernelFunctionParameter;
import jsat.parameters.MetricParameter;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * MetricKDE is a generalization of the {@link KernelDensityEstimator} to the
 * multivariate case. A {@link KernelFunction} is used to weight the
 * contribution of each data point, and a {@link DistanceMetric } is used to
 * effectively alter the shape of the kernel. The MetricKDE uses one bandwidth
 * parameter, which can be estimated using a nearest neighbor approach, or tuned
 * by hand. The bandwidth of the MetricKDE can not be estimated en the same way
 * as the univariate case.
 *
 * @author Edward Raff
 */
public class MetricKDE extends MultivariateKDE implements Parameterized {

  private static final long serialVersionUID = -2084039950938740815L;
  private static final VectorCollectionFactory<VecPaired<Vec, Integer>> defaultVCF = new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>();
  /**
   * When estimating the bandwidth, the distances of the k'th nearest neighbors
   * are used to perform the estimate. The default value of this k is
   * {@value #DEFAULT_K}
   */
  public static final int DEFAULT_K = 3;
  /**
   * When estimating the bandwidth, the distances of the k'th nearest neighbors
   * are used to perform the estimate. The default number of standard deviations
   * from the mean to add to the bandwidth estimate is
   * {@value #DEFAULT_STND_DEV}
   */
  public static final double DEFAULT_STND_DEV = 2.0;
  /**
   * When estimating the bandwidth, the distances of the k'th nearest neighbors
   * are used to perform the estimate. The weight of each neighbor is controlled
   * by the kernel function.
   */
  public static final KernelFunction DEFAULT_KF = EpanechnikovKF.getInstance();
  private KernelFunction kf;
  private double bandwidth;
  private DistanceMetric distanceMetric;

  private final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf;

  private VectorCollection<VecPaired<Vec, Integer>> vecCollection;

  private int defaultK;

  private double defaultStndDev;

  private final List<Parameter> parameters = Collections.unmodifiableList(new ArrayList<Parameter>() {
    /**
     *
     */
    private static final long serialVersionUID = -2830924861210733734L;

    {
      add(new KernelFunctionParameter() {

        private static final long serialVersionUID = 560041843101841185L;

        @Override
        public KernelFunction getObject() {
          return getKernelFunction();
        }

        @Override
        public boolean setObject(final KernelFunction obj) {
          setKernelFunction(obj);
          return true;
        }
      });

      add(new MetricParameter() {

        private static final long serialVersionUID = 1506569342529820853L;

        @Override
        public DistanceMetric getMetric() {
          return getDistanceMetric();
        }

        @Override
        public boolean setMetric(final DistanceMetric val) {
          setDistanceMetric(val);
          return true;
        }
      });

      add(new IntParameter() {

        private static final long serialVersionUID = 2109791176169136850L;

        @Override
        public String getASCIIName() {
          return "k Neighbors for Bandwidth Estimation";
        }

        @Override
        public int getValue() {
          return getDefaultK();
        }

        @Override
        public boolean setValue(final int val) {
          if (val < 1) {
            return false;
          }
          setDefaultK(val);
          return true;
        }
      });

      add(new DoubleParameter() {

        private static final long serialVersionUID = 685333554755596799L;

        @Override
        public String getASCIIName() {
          return "Standard Deviations for Bandwidth Estimation";
        }

        @Override
        public double getValue() {
          return getDefaultStndDev();
        }

        @Override
        public boolean setValue(final double val) {
          try {
            setDefaultStndDev(val);
            return true;
          } catch (final ArithmeticException e) {
            return false;
          }
        }
      });
    }
  });

  private final Map<String, Parameter> paramMap = Parameter.toParameterMap(parameters);

  /**
   * Creates a new KDE object that still needs a data set to model the
   * distribution of
   */
  public MetricKDE() {
    this(DEFAULT_KF, new EuclideanDistance(), defaultVCF);
  }

  /**
   * Creates a new KDE object that still needs a data set to model the
   * distribution of
   *
   * @param distanceMetric
   *          the distance metric to use
   */
  public MetricKDE(final DistanceMetric distanceMetric) {
    this(DEFAULT_KF, distanceMetric, defaultVCF);
  }

  /**
   * Creates a new KDE object that still needs a data set to model the
   * distribution of
   *
   * @param distanceMetric
   *          the distance metric to use
   * @param vcf
   *          a factory to generate vector collection from
   */
  public MetricKDE(final DistanceMetric distanceMetric, final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf) {
    this(DEFAULT_KF, distanceMetric, vcf);
  }

  public MetricKDE(final KernelFunction kf, final DistanceMetric distanceMetric) {
    this(kf, distanceMetric, new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>());
  }

  /**
   * Creates a new KDE object that still needs a data set to model the
   * distribution of
   *
   * @param kf
   *          the kernel function to use
   * @param distanceMetric
   *          the distance metric to use
   * @param vcf
   *          a factory to generate vector collection from
   */
  public MetricKDE(final KernelFunction kf, final DistanceMetric distanceMetric,
      final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf) {
    this(kf, distanceMetric, vcf, DEFAULT_K, DEFAULT_STND_DEV);
  }

  /**
   * Creates a new KDE object that still needs a data set to model the
   * distribution of
   *
   * @param kf
   *          the kernel function to use
   * @param distanceMetric
   *          the distance metric to use
   * @param vcf
   *          a factory to generate vector collection from
   * @param defaultK
   *          the default neighbor to use when estimating the bandwidth
   * @param defaultStndDev
   *          the default multiple of standard deviations to add when estimating
   *          the bandwidth
   */
  public MetricKDE(final KernelFunction kf, final DistanceMetric distanceMetric,
      final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf, final int defaultK, final double defaultStndDev) {
    setKernelFunction(kf);
    this.distanceMetric = distanceMetric;
    this.vcf = vcf;
    setDefaultK(defaultK);
    setDefaultStndDev(defaultStndDev);
  }

  @Override
  public MetricKDE clone() {
    final MetricKDE clone = new MetricKDE(kf, distanceMetric.clone(), vcf.clone(), defaultK, defaultStndDev);
    clone.bandwidth = bandwidth;
    if (vecCollection != null) {
      clone.vecCollection = vecCollection.clone();
    }
    return clone;
  }

  /**
   * Returns the current bandwidth used
   *
   * @return the current bandwidth
   */
  public double getBandwith() {
    return bandwidth;
  }

  /**
   * Returns the default value of the k'th nearest neighbor to use when not
   * specified.
   *
   * @return the default neighbor used to estimate the bandwidth when not
   *         specified
   */
  public int getDefaultK() {
    return defaultK;
  }

  /**
   * Returns the multiple of the standard deviations that is added to the
   * bandwidth estimate
   *
   * @return the multiple of the standard deviations that is added to the
   *         bandwidth estimate
   */
  public double getDefaultStndDev() {
    return defaultStndDev;
  }

  /**
   * Returns the distance metric that is used for density estimation
   *
   * @return the metric used
   */
  public DistanceMetric getDistanceMetric() {
    return distanceMetric;
  }

  @Override
  public KernelFunction getKernelFunction() {
    return kf;
  }

  @Override
  public List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> getNearby(final Vec x) {
    if (vecCollection == null) {
      throw new UntrainedModelException("Model has not yet been created");
    }
    final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> nearBy = getNearbyRaw(x);
    // Normalize from their distances to their weights by kernel function
    for (final VecPaired<VecPaired<Vec, Integer>, Double> result : nearBy) {
      result.setPair(kf.k(result.getPair()));
    }
    return nearBy;
  }

  @Override
  public List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> getNearbyRaw(final Vec x) {
    if (vecCollection == null) {
      throw new UntrainedModelException("Model has not yet been created");
    }
    final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> nearBy = vecCollection.search(x,
        bandwidth * kf.cutOff());

    for (final VecPaired<VecPaired<Vec, Integer>, Double> result : nearBy) {
      result.setPair(result.getPair() / bandwidth);
    }
    return nearBy;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return paramMap.get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return parameters;
  }

  @Override
  public double pdf(final Vec x) {
    final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> nearBy = getNearby(x);

    if (nearBy.isEmpty()) {
      return 0;
    }

    double PDF = 0;
    for (final VecPaired<VecPaired<Vec, Integer>, Double> result : nearBy) {
      PDF += result.getPair();
    }

    return PDF / (vecCollection.size() * Math.pow(bandwidth, nearBy.get(0).length()));
  }

  /**
   * Sampling not yet supported
   *
   * @param count
   * @param rand
   * @return will not return
   * @throws UnsupportedOperationException
   *           not yet implemented
   */
  @Override
  public List<Vec> sample(final int count, final Random rand) {
    // TODO implement sampling
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void scaleBandwidth(final double scale) {
    bandwidth *= scale;
  }

  /**
   * Sets the bandwidth used to estimate the density of the underlying
   * distribution. Too small a bandwidth results in high variance, while too
   * large causes high bias.
   *
   * @param bandwidth
   *          the bandwidth to use for estimation
   * @throws ArithmeticException
   *           if the bandwidth given is not a positive number
   */
  public void setBandwith(final double bandwidth) {
    if (bandwidth <= 0 || Double.isNaN(bandwidth) || Double.isInfinite(bandwidth)) {
      throw new ArithmeticException("Invalid bandwith given, bandwith must be a positive number, not " + bandwidth);
    }
    this.bandwidth = bandwidth;
  }

  /**
   * When estimating the bandwidth, the mean of the k'th nearest neighbors to
   * each data point is used. This value controls the default value of k used
   * when it is not specified.
   *
   * @param defaultK
   */
  public void setDefaultK(final int defaultK) {
    if (defaultK <= 0) {
      throw new ArithmeticException("At least one neighbor must be taken into acount, " + defaultK + " is invalid");
    }
    this.defaultK = defaultK;
  }

  /**
   * When estimating the bandwidth, the mean of the neighbor distances is used,
   * and a multiple of the standard deviations is added. This controls the
   * multiplier value used when the bandwidth is not specified. Zero and
   * negative multipliers are allowed, but a negative multiplier may result in
   * the fitting failing.
   *
   * @param defaultStndDev
   *          the multiple of the standard deviation to add the to bandwidth
   *          estimate
   */
  public void setDefaultStndDev(final double defaultStndDev) {
    if (Double.isInfinite(defaultStndDev) || Double.isNaN(defaultStndDev) || defaultStndDev <= 0) {
      throw new ArithmeticException(
          "The number of standard deviations to remove must bea postive number, not " + defaultStndDev);
    }
    this.defaultStndDev = defaultStndDev;
  }

  /**
   * Sets the distance metric that is used for density estimation
   *
   * @param distanceMetric
   *          the metric to use
   */
  public void setDistanceMetric(final DistanceMetric distanceMetric) {
    this.distanceMetric = distanceMetric;
  }

  public void setKernelFunction(final KernelFunction kf) {
    this.kf = kf;
  }

  @Override
  public <V extends Vec> boolean setUsingData(final List<V> dataSet) {
    return setUsingData(dataSet, defaultK);
  }

  /**
   * Sets the KDE to model the density of the given data set with the specified
   * bandwidth
   *
   * @param dataSet
   *          the data set to model the density of
   * @param bandwith
   *          the bandwidth
   * @return <tt>true</tt> if the model was fit, <tt>false</tt> if it could not
   *         be fit.
   */
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final double bandwith) {
    return setUsingData(dataSet, bandwith, null);
  }

  /**
   * Sets the KDE to model the density of the given data set with the specified
   * bandwidth
   *
   * @param dataSet
   *          the data set to model the density of
   * @param bandwith
   *          the bandwidth
   * @param threadpool
   *          the source of threads for parallel construction
   * @return <tt>true</tt> if the model was fit, <tt>false</tt> if it could not
   *         be fit.
   */
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final double bandwith,
      final ExecutorService threadpool) {
    setBandwith(bandwith);
    final List<VecPaired<Vec, Integer>> indexVectorPair = new ArrayList<VecPaired<Vec, Integer>>(dataSet.size());
    for (int i = 0; i < dataSet.size(); i++) {
      indexVectorPair.add(new VecPaired<Vec, Integer>(dataSet.get(i), i));
    }

    TrainableDistanceMetric.trainIfNeeded(distanceMetric, dataSet, threadpool);

    if (threadpool == null) {
      vecCollection = vcf.getVectorCollection(indexVectorPair, distanceMetric);
    } else {
      vecCollection = vcf.getVectorCollection(indexVectorPair, distanceMetric, threadpool);
    }

    return true;
  }

  @Override
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final ExecutorService threadpool) {
    return setUsingData(dataSet, defaultK, threadpool);
  }

  /**
   * Sets the KDE to model the density of the given data set by estimating the
   * bandwidth by using the <tt>k</tt> nearest neighbors of each data point.
   *
   * @param dataSet
   *          the data set to model the density of
   * @param k
   *          the number of neighbors to use to estimate the bandwidth
   * @return <tt>true</tt> if the model was fit, <tt>false</tt> if it could not
   *         be fit.
   */
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final int k) {
    return setUsingData(dataSet, k, defaultStndDev);
  }

  /**
   * Sets the KDE to model the density of the given data set by estimating the
   * bandwidth by using the <tt>k</tt> nearest neighbors of each data data
   * point. <br>
   * The bandwidth estimate is calculate as the mean of the distances of the
   * k'th nearest neighbor plus <tt>stndDevs</tt> standard deviations added to
   * the mean.
   *
   * @param dataSet
   *          the data set to model the density of
   * @param k
   *          the number of neighbors to use to estimate the bandwidth
   * @param stndDevs
   *          the multiple of the standard deviation to add to the mean of the
   *          distances
   * @return <tt>true</tt> if the model was fit, <tt>false</tt> if it could not
   *         be fit.
   */
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final int k, final double stndDevs) {
    return setUsingData(dataSet, k, stndDevs, null);
  }

  /**
   * Sets the KDE to model the density of the given data set by estimating the
   * bandwidth by using the <tt>k</tt> nearest neighbors of each data data
   * point. <br>
   * The bandwidth estimate is calculate as the mean of the distances of the
   * k'th nearest neighbor plus <tt>stndDevs</tt> standard deviations added to
   * the mean.
   *
   * @param dataSet
   *          the data set to model the density of
   * @param k
   *          the number of neighbors to use to estimate the bandwidth
   * @param stndDevs
   *          the multiple of the standard deviation to add to the mean of the
   *          distances
   * @param threadpool
   *          the source of threads to use for computation
   * @return <tt>true</tt> if the model was fit, <tt>false</tt> if it could not
   *         be fit.
   */
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final int k, final double stndDevs,
      final ExecutorService threadpool) {
    final List<VecPaired<Vec, Integer>> indexVectorPair = new ArrayList<VecPaired<Vec, Integer>>(dataSet.size());
    for (int i = 0; i < dataSet.size(); i++) {
      indexVectorPair.add(new VecPaired<Vec, Integer>(dataSet.get(i), i));
    }
    TrainableDistanceMetric.trainIfNeeded(distanceMetric, dataSet, threadpool);
    vecCollection = vcf.getVectorCollection(indexVectorPair, distanceMetric);

    // Take the average of the k'th neighbor distance to use as the bandwith
    OnLineStatistics stats;
    if (threadpool == null) {// k+1 b/c the first nearest neighbor will be
                             // itself
      stats = VectorCollectionUtils.getKthNeighborStats(vecCollection, dataSet, k + 1);
    } else {
      try {
        stats = VectorCollectionUtils.getKthNeighborStats(vecCollection, dataSet, k + 1, threadpool);
      } catch (final InterruptedException ex) {
        Logger.getLogger(MetricKDE.class.getName()).log(Level.SEVERE, null, ex);
        return false;
      } catch (final ExecutionException ex) {
        Logger.getLogger(MetricKDE.class.getName()).log(Level.SEVERE, null, ex);
        return false;
      }
    }

    setBandwith(stats.getMean() + stats.getStandardDeviation() * stndDevs);

    return true;
  }

  /**
   * Sets the KDE to model the density of the given data set by estimating the
   * bandwidth by using the <tt>k</tt> nearest neighbors of each data point.
   *
   * @param dataSet
   *          the data set to model the density of
   * @param k
   *          the number of neighbors to use to estimate the bandwidth
   * @param threadpool
   *          the source of threads for computation
   * @return <tt>true</tt> if the model was fit, <tt>false</tt> if it could not
   *         be fit.
   */
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final int k, final ExecutorService threadpool) {
    return setUsingData(dataSet, k, defaultStndDev, threadpool);
  }

  @Override
  public boolean setUsingDataList(final List<DataPoint> dataPoints) {
    final List<Vec> dataSet = new ArrayList<Vec>(dataPoints.size());
    for (final DataPoint dp : dataPoints) {
      dataSet.add(dp.getNumericalValues());
    }
    return setUsingData(dataSet);
  }

  @Override
  public boolean setUsingDataList(final List<DataPoint> dataPoints, final ExecutorService threadpool) {
    final List<Vec> dataSet = new ArrayList<Vec>(dataPoints.size());
    for (final DataPoint dp : dataPoints) {
      dataSet.add(dp.getNumericalValues());
    }
    return setUsingData(dataSet, threadpool);
  }

}
