/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
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
import jsat.classifiers.DataPointPair;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class PegasosTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public PegasosTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of trainC method, of class Pegasos.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");

    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final Pegasos instance = new Pegasos();
    instance.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  /**
   * Test of trainC method, of class Pegasos.
   */
  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    final ExecutorService threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final Pegasos instance = new Pegasos();
    instance.trainC(train, threadPool);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
    threadPool.shutdown();
  }

}
