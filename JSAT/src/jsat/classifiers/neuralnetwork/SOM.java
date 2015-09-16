package jsat.classifiers.neuralnetwork;

import static java.lang.Math.max;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.ExponetialDecay;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.ArrayUtils;
import jsat.utils.PairedReturn;
import jsat.utils.SystemInfo;

/**
 * An implementation of a Self Organizing Map, also called a Kohonen Map. It is
 * linked to many other algorithms, and is an unsupervised learning algorithm
 * that can perform classification. <br>
 * <br>
 * The SOM is useful for visualizing data sets, though this is not yet
 * implemented.
 *
 * @author Edward Raff
 */
public class SOM implements Classifier, Parameterized {
  // TODO add code for visualizing the SOM

  private static final long serialVersionUID = -6444988770441043797L;
  public static final int DEFAULT_MAX_ITERS = 500;
  public static final KernelFunction DEFAULT_KF = EpanechnikovKF.getInstance();
  public static final double DEFAULT_LEARNING_RATE = 0.1;
  public static final DecayRate DEFAULT_LEARNING_DECAY = new ExponetialDecay();
  public static final DecayRate DEFAULT_NEIGHBOR_DECAY = new ExponetialDecay();

  private int somWidth;
  private int somHeight;
  private int maxIters;
  private final KernelFunction kf;
  private double initialLearningRate;
  private DecayRate learningDecay;
  private DecayRate neighborDecay;
  private final DistanceMetric dm;
  private final VectorCollectionFactory<VecPaired<Vec, Integer>> vcFactory;

  private Vec[][] weights;
  private CategoricalResults[] crWeightPairs;
  private VectorCollection<VecPaired<Vec, Integer>> vcCollection;
  // Used for parallel varient
  /**
   * Contains the sum of all inputs that were the BMU for the given index. The
   * final list of data point should be a synchronized list so that multiple
   * threads can add to the list safely
   */
  private List<List<List<DataPoint>>> weightUpdates;

  /**
   * Creates a new SOM using the given parameters
   *
   * @param dm
   *          the distance metric to use when comparing points
   * @param somHeight
   *          the height of the SOM lattice
   * @param somWeight
   *          the weight of the SOM lattice
   */
  public SOM(final DistanceMetric dm, final int somHeight, final int somWeight) {
    this(dm, somHeight, somWeight, new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>());
  }

  /**
   * Creates a new SOM using the given parameters
   *
   * @param dm
   *          the distance metric to use when comparing points
   * @param somHeight
   *          the height of the SOM lattice
   * @param somWeight
   *          the weight of the SOM lattice
   * @param vcFactory
   *          the vector collection factory to use for containing points
   */
  public SOM(final DistanceMetric dm, final int somHeight, final int somWeight,
      final VectorCollectionFactory<VecPaired<Vec, Integer>> vcFactory) {
    this(DEFAULT_MAX_ITERS, DEFAULT_KF, DEFAULT_LEARNING_RATE, DEFAULT_LEARNING_DECAY, DEFAULT_NEIGHBOR_DECAY, dm,
        somHeight, somWeight, vcFactory);
  }

  /**
   * Creates a new SOM using the given parameters using the
   * {@link EuclideanDistance}
   *
   * @param somHeight
   *          the height of the SOM lattice
   * @param somWeight
   *          the weight of the SOM lattice
   */
  public SOM(final int somHeight, final int somWeight) {
    this(new EuclideanDistance(), somHeight, somWeight);
  }

  private SOM(final int maxIters, final KernelFunction kf, final double initialLearningRate,
      final DecayRate learningDecay, final DecayRate neighborDecay, final DistanceMetric dm, final int somHeight,
      final int somWeight, final VectorCollectionFactory<VecPaired<Vec, Integer>> vcFactory) {
    this.somHeight = somHeight;
    somWidth = somWeight;
    this.maxIters = maxIters;
    this.kf = kf;
    this.initialLearningRate = initialLearningRate;
    this.learningDecay = learningDecay;
    this.neighborDecay = neighborDecay;
    this.dm = dm;
    this.vcFactory = vcFactory;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (crWeightPairs == null) {
      throw new UntrainedModelException();
    }
    return crWeightPairs[vcCollection.search(data.getNumericalValues(), 1).get(0).getVector().getPair()];
  }

  @Override
  public SOM clone() {
    final SOM clone = new SOM(maxIters, kf, initialLearningRate, learningDecay, neighborDecay, dm.clone(), somHeight,
        somHeight, vcFactory.clone());
    if (weights != null) {
      clone.weights = new Vec[weights.length][weights[0].length];
      for (int i = 0; i < weights.length; i++) {
        for (int j = 0; j < weights[i].length; j++) {
          clone.weights[i][j] = weights[i][j].clone();
        }
      }
    }
    if (vcCollection != null) {
      clone.vcCollection = vcCollection.clone();
    }
    if (crWeightPairs != null) {
      clone.crWeightPairs = new CategoricalResults[crWeightPairs.length];
      for (int i = 0; i < crWeightPairs.length; i++) {
        clone.crWeightPairs[i] = crWeightPairs[i].clone();
      }
    }
    return clone;
  }

  /**
   * Finds the Best Matching Unit
   *
   * @param numericalValues
   *          the vector to find hte BMU of
   * @return the BMU of the given vector
   */
  private PairedReturn<Integer, Integer> getBMU(final Vec numericalValues) {
    double bestDist = Double.MAX_VALUE;
    int x = -1, y = -1;
    for (int i = 0; i < weights.length; i++) {
      final Vec[] weights_i = weights[i];
      for (int j = 0; j < weights[i].length; j++) {
        final double dist = dm.dist(weights_i[j], numericalValues);
        if (dist < bestDist) {
          bestDist = dist;
          x = i;
          y = j;
        }
      }
    }

    return new PairedReturn<Integer, Integer>(x, y);
  }

  /**
   * Returns the rate at which input is incorporated at each iteration of the
   * SOM
   *
   * @return the rate the SOM learns at
   */
  public double getInitialLearningRate() {
    return initialLearningRate;
  }

  /**
   * The rate the SOM learns decays over each iteration, and this defines the
   * way in which the rate decays.
   *
   * @return the decay for the learning rate
   */
  public DecayRate getLearningDecay() {
    return learningDecay;
  }

  /**
   * Returns the maximum number of iterations that will be used to converge
   *
   * @return the max iterations of the algorithm
   */
  public int getMaxIterations() {
    return maxIters;
  }

  /**
   * The range of effect each data point has decays with each iteration, and
   * this defines the way in which the rate decays.
   *
   * @return the decay for the neighbor range.
   */
  public DecayRate getNeighborDecay() {
    return neighborDecay;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Parameter.getParamsFromMethods(this);
  }

  /**
   * Returns the height of the SOM lattice to create
   *
   * @return the height of the lattice
   */
  public int getSomHeight() {
    return somHeight;
  }

  /**
   * Returns the width of the SOM lattice to create
   *
   * @return the width of the lattice
   */
  public int getSomWidth() {
    return somWidth;
  }

  /**
   *
   * @param D
   *          the dimension of the data set
   * @return the initial neighbor radius
   */
  private double intitalizeWeights(final int D) {
    // TODO random intialization is theoretical interesting, but technically
    // slower. Faster intializations exist
    for (int i = 0; i < somHeight; i++) {
      for (int j = 0; j < somWidth; j++) {
        weights[i][j] = Vec.random(D);
      }
    }
    return max(somWidth, somHeight);
  }

  private void iterationStep(final ExecutorService execServ, final int i, final DataSet dataSet, final double nbrRange,
      final double nbrRangeSqrd, final Vec scratch, final double learnRate) {
    final Vec input_i = dataSet.getDataPoint(i).getNumericalValues();
    final PairedReturn<Integer, Integer> closestBMUPR = getBMU(input_i);
    final int xBest = closestBMUPR.getFirstItem();
    final int yBest = closestBMUPR.getSecondItem();

    // The bounding square of values that need to be updated
    final int xStart = Math.max((int) (xBest - nbrRange) - 1, 0);
    final int yStart = Math.max((int) (yBest - nbrRange) - 1, 0);
    final int xEnd = Math.min((int) (xBest + nbrRange) + 1, somWidth);
    final int yEnd = Math.min((int) (yBest + nbrRange) + 1, somHeight);

    for (int x = xStart; x < xEnd; x++) {
      final Vec[] weights_x = weights[x];
      for (int y = yStart; y < yEnd; y++) {
        final int xLength = xBest - x;
        final int yLength = yBest - y;
        final int pointDistSqrd = xLength * xLength + yLength * yLength;

        if (pointDistSqrd < nbrRangeSqrd) // point is in the circle range,
        {
          final double distWeight = kf.k(sqrt(pointDistSqrd) / nbrRange);
          final Vec weights_xy = weights_x[y];
          if (execServ == null) {
            updateWeight(input_i, scratch, weights_xy, distWeight * learnRate);
          } else {
            weightUpdates.get(x).get(y).add(dataSet.getDataPoint(i));
          }
        }
      }

    }
  }

  /**
   * Sets the rate at which input is incorporated at each iteration of the SOM
   * algorithm
   *
   * @param initialLearningRate
   *          the rate the SOM learns at
   */
  public void setInitialLearningRate(final double initialLearningRate) {
    if (Double.isInfinite(initialLearningRate) || Double.isNaN(initialLearningRate) || initialLearningRate <= 0) {
      throw new ArithmeticException("Learning rate must be a positive constant, not " + initialLearningRate);
    }
    this.initialLearningRate = initialLearningRate;
  }

  /**
   * The rate the SOM learns decays over each iteration, and this defines the
   * way in which the rate decays.
   *
   * @param learningDecay
   *          the decay for the learning rate
   */
  public void setLearningDecay(final DecayRate learningDecay) {
    if (learningDecay == null) {
      throw new NullPointerException("Can not set a decay rate to null");
    }
    this.learningDecay = learningDecay;
  }

  /**
   * Sets the maximum number of iterations that will be used to converge
   *
   * @param maxIters
   *          the max iterations of the algorithm
   */
  public void setMaxIterations(final int maxIters) {
    if (maxIters < 1) {
      throw new ArithmeticException("At least one iteration must be performed");
    }
    this.maxIters = maxIters;
  }

  /**
   * The range of effect each data point has decays with each iteration, and
   * this defines the way in which the rate decays.
   *
   * @param neighborDecay
   *          the decay for the neighbor range.
   */
  public void setNeighborDecay(final DecayRate neighborDecay) {
    if (neighborDecay == null) {
      throw new NullPointerException("Can not set a decay rate to null");
    }
    this.neighborDecay = neighborDecay;
  }

  /**
   * Sets the height of the SOM lattice to create
   *
   * @param somHeight
   *          the height of the lattice
   */
  public void setSomHeight(final int somHeight) {
    if (somHeight < 1) {
      throw new ArithmeticException("ALttice height must be positive, not " + somHeight);
    }
    this.somHeight = somHeight;
  }

  /**
   * Sets the width of the SOM lattice to create
   *
   * @param somWidth
   *          the width of the lattice
   */
  public void setSomWidth(final int somWidth) {
    if (somWidth < 1) {
      throw new ArithmeticException("Lattice width must be positive, not " + somWidth);
    }
    this.somWidth = somWidth;
  }

  private List<VecPaired<Vec, Integer>> setUpVectorCollection(final ExecutorService threadPool) {
    final List<VecPaired<Vec, Integer>> vecList = new ArrayList<VecPaired<Vec, Integer>>(somWidth * somHeight);
    for (final Vec[] weight : weights) {
      for (final Vec element : weight) {
        vecList.add(new VecPaired<Vec, Integer>(element, vecList.size()));
      }
    }
    if (threadPool == null) {
      vcCollection = vcFactory.getVectorCollection(vecList, dm);
    } else {
      vcCollection = vcFactory.getVectorCollection(vecList, dm, threadPool);
    }
    return vecList;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, null);

  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    try {
      trainSOM(dataSet, threadPool);
      final List<VecPaired<Vec, Integer>> vecList = setUpVectorCollection(threadPool);
      crWeightPairs = new CategoricalResults[vecList.size()];

      for (int i = 0; i < crWeightPairs.length; i++) {
        crWeightPairs[i] = new CategoricalResults(dataSet.getClassSize());
      }

      for (int i = 0; i < dataSet.getSampleSize(); i++) {
        final DataPoint dp = dataSet.getDataPoint(i);

        // Single nearest neighbor is the BMU
        final VecPaired<Vec, Integer> vpBMU = vcCollection.search(dp.getNumericalValues(), 1).get(0).getVector();

        final int index = vpBMU.getPair();

        crWeightPairs[index].incProb(dataSet.getDataPointCategory(i), dp.getWeight());
      }

      for (final CategoricalResults crWeightPair : crWeightPairs) {
        crWeightPair.normalize();
      }
    } catch (final InterruptedException ex) {
      Logger.getLogger(SOM.class.getName()).log(Level.SEVERE, null, ex);
    }

  }

  private void trainSOM(final DataSet dataSet, final ExecutorService execServ) throws InterruptedException {
    final int D = dataSet.getNumNumericalVars();
    weights = new Vec[somHeight][somWidth];

    final double neighborRadius = intitalizeWeights(D);

    final Random rand = new Random();

    final Vec scratch = new DenseVector(D);
    /**
     * this array is used to access the data in a random order to improve
     * convergence
     */
    final int[] pointAccessOrder = new int[dataSet.getSampleSize()];
    for (int i = 0; i < pointAccessOrder.length; i++) {
      pointAccessOrder[i] = i;
    }

    final ThreadLocal<Vec> localScratch1;
    final ThreadLocal<Vec> localScratch2;

    if (execServ != null) // Create parallel structures
    {
      weightUpdates = new ArrayList<List<List<DataPoint>>>(somHeight);

      for (int i = 0; i < somHeight; i++) {
        final ArrayList<List<DataPoint>> subList = new ArrayList<List<DataPoint>>(somWidth);
        weightUpdates.add(subList);
        for (int j = 0; j < somWidth; j++) {
          subList.add(Collections.synchronizedList(new ArrayList<DataPoint>()));
        }
      }

      localScratch1 = new ThreadLocal<Vec>() {
        @Override
        protected Vec initialValue() {
          return new DenseVector(D);
        }
      };
      localScratch2 = new ThreadLocal<Vec>() {
        @Override
        protected Vec initialValue() {
          return new DenseVector(D);
        }
      };
    } else {
      localScratch2 = localScratch1 = null;
    }

    for (int iter = 0; iter < maxIters; iter++) {
      final double nbrRange = neighborDecay.rate(iter, maxIters, neighborRadius);
      final double nbrRangeSqrd = nbrRange * nbrRange;

      final double learnRate = learningDecay.rate(iter, maxIters, initialLearningRate);

      // Set up before data loop. Shuffle for better convergence if single
      // threaded, create result queus for paralllel collection
      if (execServ == null) {
        ArrayUtils.shuffle(pointAccessOrder, rand);
      } else// Prep parallel structures
      {
        for (int i = 0; i < somHeight; i++) {
          for (int j = 0; j < somWidth; j++) {
            weightUpdates.get(i).get(j).clear();
          }
        }
      }

      // Performe main loop over all data points
      if (execServ == null) {
        for (final int element : pointAccessOrder) {
          iterationStep(execServ, element, dataSet, nbrRange, nbrRangeSqrd, scratch, learnRate);
        }
      } else// parallel
      {
        int pos = 0;
        final int size = dataSet.getSampleSize() / SystemInfo.LogicalCores;
        int extra = dataSet.getSampleSize() % SystemInfo.LogicalCores;
        final CountDownLatch cdl = new CountDownLatch(SystemInfo.LogicalCores);

        while (pos < dataSet.getSampleSize()) {
          final int to = (extra-- > 0 ? 1 : 0) + pos + size;
          final int start = pos;
          pos = to;
          execServ.submit(new Runnable() {

            @Override
            public void run() {
              for (int i = start; i < to; i++) {
                iterationStep(execServ, i, dataSet, nbrRange, nbrRangeSqrd, localScratch1.get(), learnRate);
              }
              cdl.countDown();
            }
          });
        }

        cdl.await();
      }

      // Collect results if we did parallel computation
      if (execServ != null) // Apply changes
      {
        final CountDownLatch cdl = new CountDownLatch(somHeight * somWidth);
        for (int i = 0; i < somHeight; i++) {
          for (int j = 0; j < somWidth; j++) {
            final List<DataPoint> dataList = weightUpdates.get(i).get(j);
            final int x = i, y = j;

            execServ.submit(new Runnable() {

              @Override
              public void run() {
                final Vec mean = localScratch1.get();
                mean.zeroOut();

                double denom = 0.0;
                for (final DataPoint dp : dataList) {
                  denom += dp.getWeight();
                  mean.mutableAdd(dp.getWeight(), dp.getNumericalValues());
                }

                mean.mutableDivide(denom);

                updateWeight(mean, localScratch2.get(), weights[x][y], learnRate);

                cdl.countDown();
              }
            });
          }
        }
        cdl.await();
      }
    }
  }

  private void updateWeight(final Vec input_i, final Vec scratch, final Vec weightVec, final double scale) {
    input_i.copyTo(scratch);
    scratch.mutableSubtract(weightVec);
    weightVec.mutableAdd(scale, scratch);
  }

}
