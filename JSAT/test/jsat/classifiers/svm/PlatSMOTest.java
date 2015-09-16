package jsat.classifiers.svm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.kernels.LinearKernel;
import jsat.distributions.kernels.RBFKernel;
import jsat.regression.RegressionDataSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class PlatSMOTest {

  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public PlatSMOTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of train method, of class PlatSMO.
   */
  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    final RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
    final RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));

    for (final boolean modification1 : new boolean[] { true, false }) {
      for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
        final PlatSMO smo = new PlatSMO(new RBFKernel(0.5));
        smo.setCacheMode(cacheMode);
        smo.setC(1);
        smo.setEpsilon(0.1);
        smo.setModificationOne(modification1);
        smo.train(trainSet);
        double errors = 0;
        for (int i = 0; i < testSet.getSampleSize(); i++) {
          errors += Math.pow(testSet.getTargetValue(i) - smo.regress(testSet.getDataPoint(i)), 2);
        }
        assertTrue(errors / testSet.getSampleSize() < 1);
      }
    }
  }

  /**
   * Test of train method, of class PlatSMO.
   */
  @Test
  public void testTrain_RegressionDataSet_ExecutorService() {
    System.out.println("train");
    final RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
    final RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));

    for (final boolean modification1 : new boolean[] { true, false }) {
      for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
        final PlatSMO smo = new PlatSMO(new RBFKernel(0.5));
        smo.setCacheMode(cacheMode);
        smo.setC(1);
        smo.setEpsilon(0.1);
        smo.setModificationOne(modification1);
        smo.train(trainSet, ex);
        double errors = 0;
        for (int i = 0; i < testSet.getSampleSize(); i++) {
          errors += Math.pow(testSet.getTargetValue(i) - smo.regress(testSet.getDataPoint(i)), 2);
        }
        assertTrue(errors / testSet.getSampleSize() < 1);
      }
    }
  }

  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
    final ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

    for (final boolean modification1 : new boolean[] { true, false }) {
      for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
        final PlatSMO classifier = new PlatSMO(new RBFKernel(0.5));
        classifier.setCacheMode(cacheMode);
        classifier.setC(10);
        classifier.setModificationOne(modification1);
        classifier.trainC(trainSet);
        for (int i = 0; i < testSet.getSampleSize(); i++) {
          assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }
      }
    }
  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    final ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
    final ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

    for (final boolean modification1 : new boolean[] { true, false }) {
      for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
        final PlatSMO classifier = new PlatSMO(new RBFKernel(0.5));
        classifier.setCacheMode(cacheMode);
        classifier.setC(10);
        classifier.setModificationOne(modification1);
        classifier.trainC(trainSet, ex);
        for (int i = 0; i < testSet.getSampleSize(); i++) {
          assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }
      }
    }
  }

  @Test()
  public void testTrainWarmCFastOther() {
    final ClassificationDataSet train = FixedProblems.getHalfCircles(250, new XORWOW(), 0.1, 0.2);

    final DCDs warmModel = new DCDs();
    warmModel.setUseL1(true);
    warmModel.setUseBias(true);
    warmModel.trainC(train);

    final PlatSMO warm = new PlatSMO(new LinearKernel(1));
    warm.setC(1e4);// too large to train efficently like noraml

    long start, end;

    start = System.currentTimeMillis();
    warm.trainC(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    final PlatSMO notWarm = new PlatSMO(new LinearKernel(1));
    notWarm.setC(1e4);// too large to train efficently like noraml

    start = System.currentTimeMillis();
    notWarm.trainC(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    assertTrue(warmTime < normTime * 0.75);

  }

  @Test()
  public void testTrainWarmCFastSMO() {
    // problem needs to be non-linear to make SMO work harder
    final ClassificationDataSet train = FixedProblems.getHalfCircles(250, new XORWOW(), 0.1, 0.2);

    final PlatSMO warmModel = new PlatSMO(new LinearKernel(1));
    warmModel.setC(1);
    warmModel.trainC(train);

    final PlatSMO warm = new PlatSMO(new LinearKernel(1));
    warm.setC(1e4);// too large to train efficently like noraml

    long start, end;

    start = System.currentTimeMillis();
    warm.trainC(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    final PlatSMO notWarm = new PlatSMO(new LinearKernel(1));
    notWarm.setC(1e4);// too large to train efficently like noraml

    start = System.currentTimeMillis();
    notWarm.trainC(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    assertTrue(warmTime < normTime * 0.75);

  }

  @Test()
  public void testTrainWarmRFastOther() {
    final RegressionDataSet train = FixedProblems.getLinearRegression(1000, new XORWOW());
    final double eps = train.getTargetValues().mean() / 20;

    final DCDs warmModel = new DCDs();
    warmModel.setEps(eps);
    warmModel.setUseL1(true);
    warmModel.setUseBias(true);
    warmModel.train(train);

    long start, end;

    final PlatSMO notWarm = new PlatSMO(new LinearKernel(1));
    notWarm.setEpsilon(eps);
    notWarm.setC(1e2);

    start = System.currentTimeMillis();
    notWarm.train(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    final PlatSMO warm = new PlatSMO(new LinearKernel(1));
    warm.setEpsilon(eps);
    warm.setC(1e2);

    start = System.currentTimeMillis();
    warm.train(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    assertTrue(warmTime < normTime * 0.75);

  }
}
