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
public class ROMMATest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public ROMMATest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of supportsWeightedData method, of class ROMMA.
   */
  @Test
  public void testTrain_C() {
    System.out.println("supportsWeightedData");
    final ROMMA nonAggro = new ROMMA();
    final ROMMA aggro = new ROMMA();
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    nonAggro.setEpochs(1);
    nonAggro.trainC(train);

    aggro.setEpochs(1);
    aggro.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), aggro.classify(dpp.getDataPoint()).mostLikely());
    }

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), nonAggro.classify(dpp.getDataPoint()).mostLikely());
    }
  }
}
