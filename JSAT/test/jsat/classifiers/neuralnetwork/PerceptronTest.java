/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.neuralnetwork;

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
public class PerceptronTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public PerceptronTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of trainC method, of class Perceptron.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    Perceptron instance = new Perceptron();
    instance = instance.clone();
    instance.trainC(train);
    instance = instance.clone();

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  /**
   * Test of trainC method, of class Perceptron.
   */
  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    final ExecutorService threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    Perceptron instance = new Perceptron();
    instance = instance.clone();
    instance.trainC(train, threadPool);
    instance = instance.clone();

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
    threadPool.shutdown();
  }

  /**
   * Test of trainCOnline method, of class Perceptron.
   */
  @Test
  public void testTrainCOnline() {
    System.out.println("trainCOnline");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    Perceptron instance = new Perceptron();
    instance = instance.clone();
    instance.trainCOnline(train);
    instance = instance.clone();

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
  }

}
