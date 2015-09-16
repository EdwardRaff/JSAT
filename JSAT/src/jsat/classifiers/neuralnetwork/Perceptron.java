package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.PairedReturn;

/**
 * The perceptron is a simple algorithm that attempts to find a hyperplane that
 * separates two classes. It may find any possible separating plane, and there
 * are no guarantees when the data is not linearly separable. <br>
 * It is equivalent to a single node Neural Network, and is related to SVMs
 *
 *
 * @author Edward Raff
 */
public class Perceptron implements BinaryScoreClassifier, SingleWeightVectorModel {

  /**
   * We use the probability match object to return both the vector and the bias
   * term. The first index in the double will contain the change in bias, the
   * 2nd will contain the change in global error
   */
  private class BatchTrainingUnit implements Callable<PairedReturn<Vec, Double[]>> {

    // this will be updated incrementally
    private final Vec tmpSummedErrors;
    private double biasChange;
    private double globalError;

    List<DataPointPair<Integer>> dataPoints;

    public BatchTrainingUnit(final List<DataPointPair<Integer>> toOperateOn) {
      tmpSummedErrors = new DenseVector(weights.length());
      dataPoints = toOperateOn;
      globalError = 0;
      biasChange = 0;
    }

    @Override
    public PairedReturn<Vec, Double[]> call() throws Exception {
      for (final DataPointPair<Integer> dpp : dataPoints) {

        final int output = output(dpp.getDataPoint());
        final double localError = dpp.getPair() - output;

        if (localError != 0) {// Update the weight vecotrs
          // The weight of this sample, take it into account!
          final double extraWeight = dpp.getDataPoint().getWeight();

          final double magnitude = learningRate * localError * extraWeight;

          tmpSummedErrors.mutableAdd(magnitude, dpp.getVector());
          biasChange += magnitude;
          globalError += Math.abs(localError) * extraWeight;
        }
      }

      return new PairedReturn<Vec, Double[]>(tmpSummedErrors, new Double[] { biasChange, globalError });
    }
  }

  private static final long serialVersionUID = -3605237847981632021L;
  private final double learningRate;
  private double bias;
  private Vec weights;

  private final int iteratinLimit;

  /**
   * Creates a new Perceptron learner
   */
  public Perceptron() {
    this(0.1, 400);
  }

  /**
   * Creates a new Perceptron learner
   *
   * @param learningRate
   *          the rate at which to incorporate the change of errors into the
   *          model
   * @param iteratinLimit
   *          the maximum number of iterations to perform when converging
   */
  public Perceptron(final double learningRate, final int iteratinLimit) {
    if (learningRate <= 0 || learningRate > 1) {
      throw new RuntimeException("Preceptron learning rate must be in the range (0,1]");
    }
    this.learningRate = learningRate;
    this.iteratinLimit = iteratinLimit;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(2);
    cr.setProb(output(data), 1);

    return cr;
  }

  @Override
  public Perceptron clone() {
    final Perceptron copy = new Perceptron(learningRate, iteratinLimit);
    if (weights != null) {
      copy.weights = weights.clone();
    }
    copy.bias = bias;

    return copy;
  }

  @Override
  public double getBias() {
    return bias;
  }

  @Override
  public double getBias(final int index) {
    if (index < 1) {
      return getBias();
    } else {
      throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
  }

  @Override
  public Vec getRawWeight() {
    return weights;
  }

  @Override
  public Vec getRawWeight(final int index) {
    if (index < 1) {
      return getRawWeight();
    } else {
      throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
  }

  @Override
  public double getScore(final DataPoint dp) {
    return weights.dot(dp.getNumericalValues()) + bias;
  }

  @Override
  public int numWeightsVecs() {
    return 1;
  }

  private int output(final DataPoint input) {
    final double dot = getScore(input);

    return dot >= 0 ? 1 : 0;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainCOnline(dataSet);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    if (dataSet.getClassSize() != 2) {
      throw new FailedToFitException("Preceptron only supports binary calssification");
    } else if (dataSet.getNumCategoricalVars() != 0) {
      throw new FailedToFitException("Preceptron only supports vector classification");
    }

    final List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
    Collections.shuffle(dataPoints);

    final int partions = Runtime.getRuntime().availableProcessors();

    final Random r = new Random();
    final int numerVars = dataSet.getNumNumericalVars();

    weights = new DenseVector(numerVars);
    for (int i = 0; i < weights.length(); i++) {
      // give all variables a random weight in the range [0,1]
      weights.set(i, r.nextDouble());
    }

    Vec bestWeightsSoFar = null;
    double lowestErrorSoFar = Double.MAX_VALUE;
    int iterations = 0;
    bias = 0;
    double globalError;
    do {
      globalError = 0;
      final Vec sumedErrors = new DenseVector(weights.length());
      double biasChange = 0;

      // Where our intermediate partial results will be stored
      final List<Future<PairedReturn<Vec, Double[]>>> futures = new ArrayList<Future<PairedReturn<Vec, Double[]>>>(
          partions);
      // create a task for each thing being submitied
      final int blockSize = dataPoints.size() / partions;
      for (int i = 0; i < partions; i++) {
        List<DataPointPair<Integer>> subList;
        if (i == partions - 1) {
          subList = dataPoints.subList(i * blockSize, dataPoints.size());
        } else {
          subList = dataPoints.subList(i * blockSize, (i + 1) * blockSize);
        }

        futures.add(threadPool.submit(new BatchTrainingUnit(subList)));
      }

      // Now collect the results
      for (final Future<PairedReturn<Vec, Double[]>> future : futures) {
        try {
          final PairedReturn<Vec, Double[]> partialResult = future.get();
          sumedErrors.mutableAdd(partialResult.getFirstItem());
          biasChange += partialResult.getSecondItem()[0];
          globalError += partialResult.getSecondItem()[1];
        } catch (final InterruptedException ex) {

        } catch (final ExecutionException ex) {

        }
      }

      if (globalError < lowestErrorSoFar) {
        bestWeightsSoFar = weights;
        lowestErrorSoFar = globalError;
      }

      bias += biasChange;
      weights.mutableAdd(sumedErrors);

      iterations++;
    } while (globalError > 0 && iterations < iteratinLimit);

    weights = bestWeightsSoFar;
  }

  // Uses the online training algorithm instead of the batch one.
  public void trainCOnline(final ClassificationDataSet dataSet) {
    if (dataSet.getClassSize() != 2) {
      throw new FailedToFitException("Preceptron only supports binary calssification");
    } else if (dataSet.getNumCategoricalVars() != 0) {
      throw new FailedToFitException("Preceptron only supports vector classification");
    }

    final List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
    Collections.shuffle(dataPoints);

    final Random r = new Random();
    final int numerVars = dataSet.getNumNumericalVars();

    weights = new DenseVector(numerVars);
    for (int i = 0; i < weights.length(); i++) {
      // give all variables a random weight in the range [0,1]
      weights.set(i, r.nextDouble());
    }

    Vec bestWeightsSoFar = null;
    double lowestErrorSoFar = Double.MAX_VALUE;
    int iterations = 0;

    double globalError;
    do {
      globalError = 0;
      // For each data point
      for (final DataPointPair<Integer> dpp : dataPoints) {
        final int output = output(dpp.getDataPoint());
        final double localError = dpp.getPair() - output;

        if (localError != 0) {// Update the weight vecotrs
          // The weight of this sample, take it into account!
          final double extraWeight = dpp.getDataPoint().getWeight();

          final double magnitude = learningRate * localError * extraWeight;

          weights.mutableAdd(magnitude, dpp.getVector());
          bias += magnitude;
          globalError += Math.abs(localError) * extraWeight;
        }
      }

      if (globalError < lowestErrorSoFar) {
        bestWeightsSoFar = weights;
        lowestErrorSoFar = globalError;
      }
      iterations++;
    } while (globalError > 0 && iterations < iteratinLimit);

    weights = bestWeightsSoFar;
  }
}
