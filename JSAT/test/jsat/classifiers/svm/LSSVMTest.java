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
import jsat.distributions.kernels.RBFKernel;
import jsat.regression.RegressionDataSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class LSSVMTest {

  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public LSSVMTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of train method, of class LSSVM.
   */
  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    final RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
    final RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));

    for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
      final LSSVM lssvm = new LSSVM(new RBFKernel(0.5), cacheMode);
      lssvm.setCacheMode(cacheMode);
      lssvm.setC(1);
      lssvm.train(trainSet);

      double errors = 0;
      for (int i = 0; i < testSet.getSampleSize(); i++) {
        errors += Math.pow(testSet.getTargetValue(i) - lssvm.regress(testSet.getDataPoint(i)), 2);
      }
      assertTrue(errors / testSet.getSampleSize() < 1);
    }
  }

  /**
   * Test of train method, of class LSSVM.
   */
  @Test
  public void testTrain_RegressionDataSet_ExecutorService() {
    System.out.println("train");
    final RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
    final RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));

    for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
      final LSSVM lssvm = new LSSVM(new RBFKernel(0.5), cacheMode);
      lssvm.setCacheMode(cacheMode);
      lssvm.setC(1);
      lssvm.train(trainSet, ex);

      double errors = 0;
      for (int i = 0; i < testSet.getSampleSize(); i++) {
        errors += Math.pow(testSet.getTargetValue(i) - lssvm.regress(testSet.getDataPoint(i)), 2);
      }
      assertTrue(errors / testSet.getSampleSize() < 1);
    }
  }

  /**
   * Test of trainC method, of class LSSVM.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
    final ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

    for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
      final LSSVM classifier = new LSSVM(new RBFKernel(0.5), cacheMode);
      classifier.setCacheMode(cacheMode);
      classifier.setC(1);
      classifier.trainC(trainSet);

      for (int i = 0; i < testSet.getSampleSize(); i++) {
        assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
      }
    }
  }

  /**
   * Test of trainC method, of class LSSVM.
   */
  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    final ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
    final ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

    for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
      final LSSVM classifier = new LSSVM(new RBFKernel(0.5), cacheMode);
      classifier.setCacheMode(cacheMode);
      classifier.setC(1);
      classifier.trainC(trainSet, ex);

      for (int i = 0; i < testSet.getSampleSize(); i++) {
        assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
      }
    }
  }

  @Test()
  public void testTrainWarmC() {
    final ClassificationDataSet train = FixedProblems.getHalfCircles(100, new XORWOW(), 0.1, 0.2);

    final LSSVM warmModel = new LSSVM();
    warmModel.setC(1);
    warmModel.setCacheMode(SupportVectorLearner.CacheMode.FULL);
    warmModel.trainC(train);

    final LSSVM warm = new LSSVM();
    warm.setC(2e1);
    warm.setCacheMode(SupportVectorLearner.CacheMode.FULL);

    long start, end;

    start = System.currentTimeMillis();
    warm.trainC(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    final LSSVM notWarm = new LSSVM();
    notWarm.setC(2e1);
    notWarm.setCacheMode(SupportVectorLearner.CacheMode.FULL);

    start = System.currentTimeMillis();
    notWarm.trainC(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    assertTrue(warmTime < normTime * 0.75);

  }

  @Test()
  public void testTrainWarmR() {
    final RegressionDataSet train = FixedProblems.getSimpleRegression1(75, new XORWOW());

    final LSSVM warmModel = new LSSVM();
    warmModel.setC(1);
    warmModel.setCacheMode(SupportVectorLearner.CacheMode.FULL);
    warmModel.train(train);

    final LSSVM warm = new LSSVM();
    warm.setC(1e1);
    warm.setCacheMode(SupportVectorLearner.CacheMode.FULL);

    long start, end;

    start = System.currentTimeMillis();
    warm.train(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    final LSSVM notWarm = new LSSVM();
    notWarm.setC(1e1);
    notWarm.setCacheMode(SupportVectorLearner.CacheMode.FULL);

    start = System.currentTimeMillis();
    notWarm.train(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    assertTrue(warmTime < normTime * 0.75);

  }

}
