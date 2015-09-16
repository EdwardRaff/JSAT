package jsat.classifiers.linear;

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
import jsat.classifiers.DataPointPair;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class LogisticRegressionDCDTest {

  static ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public LogisticRegressionDCDTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of trainC method, of class LogisticRegressionDCD.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final LogisticRegressionDCD lr = new LogisticRegressionDCD();
    lr.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), lr.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  /**
   * Test of trainC method, of class LogisticRegressionDCD.
   */
  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final LogisticRegressionDCD lr = new LogisticRegressionDCD();
    lr.trainC(train, ex);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), lr.classify(dpp.getDataPoint()).mostLikely());
    }
  }

}
