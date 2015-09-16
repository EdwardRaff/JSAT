/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear;

import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;

/**
 *
 * @author Edward Raff
 */
public class ALMA2Test {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public ALMA2Test() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of classify method, of class ALMA2.
   */
  @Test
  public void testTrain_C() {
    System.out.println("classify");

    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final ALMA2 alma = new ALMA2();
    alma.setEpochs(1);

    alma.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), alma.classify(dpp.getDataPoint()).mostLikely());
    }

  }

}
