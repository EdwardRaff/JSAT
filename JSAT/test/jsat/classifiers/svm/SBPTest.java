package jsat.classifiers.svm;

import static org.junit.Assert.assertEquals;

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
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class SBPTest {

  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public SBPTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
    final ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

    for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
      final SBP classifier = new SBP(new RBFKernel(0.5), cacheMode, trainSet.getSampleSize(), 0.01);
      classifier.trainC(trainSet);

      for (int i = 0; i < testSet.getSampleSize(); i++) {
        assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
      }
    }
  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    final ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
    final ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

    for (final SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values()) {
      final SBP classifier = new SBP(new RBFKernel(0.5), cacheMode, trainSet.getSampleSize(), 0.01);
      classifier.trainC(trainSet, ex);

      for (int i = 0; i < testSet.getSampleSize(); i++) {
        assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
      }
    }
  }
}
