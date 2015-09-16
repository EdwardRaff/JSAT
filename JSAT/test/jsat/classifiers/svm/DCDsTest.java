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
import jsat.classifiers.DataPointPair;
import jsat.regression.RegressionDataSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class DCDsTest {

  static private ExecutorService threadPool;

  @BeforeClass
  public static void setUpClass() {
    threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    threadPool.shutdown();
  }

  public DCDsTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    final Random rand = new Random();

    final DCDs dcds = new DCDs();
    dcds.train(FixedProblems.getLinearRegression(400, rand));

    for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList()) {
      final double truth = dpp.getPair();
      final double pred = dcds.regress(dpp.getDataPoint());

      final double relErr = (truth - pred) / truth;
      assertEquals(0.0, relErr, 0.1);// Give it a decent wiggle room b/c of
                                     // regularization
    }
  }

  @Test
  public void testTrain_RegressionDataSet_ExecutorService() {
    System.out.println("train");
    final Random rand = new Random();

    final DCDs dcds = new DCDs();
    dcds.train(FixedProblems.getLinearRegression(400, rand), threadPool);

    for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList()) {
      final double truth = dpp.getPair();
      final double pred = dcds.regress(dpp.getDataPoint());

      final double relErr = (truth - pred) / truth;
      assertEquals(0.0, relErr, 0.1);// Give it a decent wiggle room b/c of
                                     // regularization
    }
  }

  /**
   * Test of trainC method, of class DCDs.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final DCDs instance = new DCDs();
    instance.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  /**
   * Test of trainC method, of class DCDs.
   */
  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");

    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final DCDs instance = new DCDs();
    instance.trainC(train, threadPool);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  @Test()
  public void testTrainWarmC() {
    final ClassificationDataSet train = FixedProblems.getHalfCircles(10000, new XORWOW(), 0.1, 0.5);

    final DCDs warmModel = new DCDs();
    warmModel.trainC(train);
    warmModel.setC(1);

    long start, end;

    final DCDs notWarm = new DCDs();
    notWarm.setC(1e1);

    start = System.currentTimeMillis();
    notWarm.trainC(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    final DCDs warm = new DCDs();
    warm.setC(1e1);

    start = System.currentTimeMillis();
    warm.trainC(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    assertTrue(warmTime < normTime * 0.80);
  }

  @Test()
  public void testTrainWarR() {
    final RegressionDataSet train = FixedProblems.getSimpleRegression1(4000, new XORWOW());
    final double eps = train.getTargetValues().mean() / 0.9;

    final DCDs warmModel = new DCDs();
    warmModel.setEps(eps);
    warmModel.train(train);

    final DCDs warm = new DCDs();
    warm.setEps(eps);
    warm.setC(1e1);// too large to train efficently like noraml

    long start, end;

    start = System.currentTimeMillis();
    warm.train(train, warmModel);
    end = System.currentTimeMillis();
    final long warmTime = end - start;

    final DCDs notWarm = new DCDs();
    notWarm.setEps(eps);
    notWarm.setC(1e1);// too large to train efficently like noraml

    start = System.currentTimeMillis();
    notWarm.train(train);
    end = System.currentTimeMillis();
    final long normTime = end - start;

    assertTrue(warmTime < normTime * 0.80);
  }
}
