package jsat.classifiers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;

import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DenseSparseMetric;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.FakeExecutor;

/**
 *
 * @author Edward Raff
 */
public class Rocchio implements Classifier {

  /*
   * Runnable that will sum all the vectors added until it is done
   *
   */
  private class RocchioAdder implements Runnable {

    double weightSum;

    final CountDownLatch latch;

    final Vec rocchioVec;
    final List<DataPoint> input;
    final int index;

    public RocchioAdder(final CountDownLatch latch, final int index, final Vec rocchioVec,
        final List<DataPoint> input) {
      this.latch = latch;
      this.index = index;
      this.rocchioVec = rocchioVec;
      this.input = input;
      weightSum = 0;
    }

    @Override
    public void run() {
      for (final DataPoint dp : input) {
        final double w = dp.getWeight();
        final Vec v = dp.getNumericalValues();
        weightSum += w;
        rocchioVec.mutableAdd(w, v);
      }

      rocchioVec.mutableDivide(weightSum);
      if (dsdm != null) {
        summaryConsts[index] = dsdm.getVectorConstant(rocchioVec);
      }
      latch.countDown();
    }

  }

  private static final long serialVersionUID = 889524967453326517L;
  private List<Vec> rocVecs;
  private final DistanceMetric dm;
  private final DenseSparseMetric dsdm;

  private double[] summaryConsts;

  public Rocchio() {
    this(new EuclideanDistance());
  }

  public Rocchio(final DistanceMetric dm) {
    this.dm = dm;
    dsdm = dm instanceof DenseSparseMetric ? (DenseSparseMetric) dm : null;
    rocVecs = null;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(rocVecs.size());
    double sum = 0;

    final Vec target = data.getNumericalValues();

    // Record the average for each class
    for (int i = 0; i < rocVecs.size(); i++) {
      double distance;
      if (summaryConsts == null) {
        distance = dm.dist(rocVecs.get(i), target);
      } else {
        distance = dsdm.dist(summaryConsts[i], rocVecs.get(i), target);
      }
      sum += distance;
      cr.setProb(i, distance);
    }

    // now scale, set them all to 1-distance/sumOfDistances. We will call that
    // out probablity
    for (int i = 0; i < rocVecs.size(); i++) {
      cr.setProb(i, 1.0 - cr.getProb(i) / sum);
    }

    return cr;
  }

  @Override
  public Rocchio clone() {
    final Rocchio copy = new Rocchio(dm);
    if (rocVecs != null) {
      copy.rocVecs = new ArrayList<Vec>(rocVecs.size());
      for (final Vec v : rocVecs) {
        copy.rocVecs.add(v.clone());
      }
    }
    if (summaryConsts != null) {
      copy.summaryConsts = Arrays.copyOf(summaryConsts, summaryConsts.length);
    }
    return copy;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, new FakeExecutor());
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    if (dataSet.getNumCategoricalVars() != 0) {
      throw new FailedToFitException("Classifier requires all variables be numerical");
    }
    final int N = dataSet.getClassSize();
    rocVecs = new ArrayList<Vec>(N);

    TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadPool);

    // dimensions
    final int d = dataSet.getNumNumericalVars();

    summaryConsts = new double[d];

    // Set up a bunch of threads to add vectors together in the background
    final CountDownLatch cdl = new CountDownLatch(N);
    for (int i = 0; i < N; i++) {
      final Vec rochVec = new DenseVector(d);
      rocVecs.add(rochVec);

      threadPool.submit(new RocchioAdder(cdl, i, rochVec, dataSet.getSamples(i)));
    }

    try {
      cdl.await();
    } catch (final InterruptedException ex) {
    }

  }

}
