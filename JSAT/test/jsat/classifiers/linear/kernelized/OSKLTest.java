package jsat.classifiers.linear.kernelized;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class OSKLTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public OSKLTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testClone() {
    System.out.println("clone");

    OSKL instance = new OSKL(new RBFKernel(0.5), 10);

    final ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, new XORWOW());
    final ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, new XORWOW(), 2.0, 10.0);

    instance = instance.clone();

    instance.trainC(t1);

    instance.setUseAverageModel(true);
    final OSKL result = instance.clone();
    assertTrue(result.isUseAverageModel());
    for (int i = 0; i < t1.getSampleSize(); i++) {
      assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
    }
    result.trainC(t2);

    for (int i = 0; i < t1.getSampleSize(); i++) {
      assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
    }

    for (int i = 0; i < t2.getSampleSize(); i++) {
      assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
    }

  }

  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");

    for (final boolean useAverageModel : new boolean[] { true, false }) {
      for (final int burnin : new int[] { 0, 50, 100, 250 }) {
        final OSKL instance = new OSKL(new RBFKernel(0.5), 1.5);
        instance.setBurnIn(burnin);
        instance.setUseAverageModel(useAverageModel);

        final ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, new XORWOW());
        final ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, new XORWOW());

        final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);

        assertEquals(0, cme.getErrorRate(), 0.0);
      }
    }

  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");

    final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

    for (final boolean useAverageModel : new boolean[] { true, false }) {
      for (final int burnin : new int[] { 0, 50, 100, 250 }) {
        final OSKL instance = new OSKL(new RBFKernel(0.5), 1.5);
        instance.setBurnIn(burnin);
        instance.setUseAverageModel(useAverageModel);

        final ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, new XORWOW());
        final ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, new XORWOW());

        final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);

        assertEquals(0, cme.getErrorRate(), 0.0);
      }
    }
    ex.shutdownNow();

  }
}
