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
public class SCWTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public SCWTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testTrainC_Diag() {
    System.out.println("TrainC_Diag");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final SCW.Mode mode : SCW.Mode.values()) {
      final SCW scwDiag = new SCW(0.9, mode, true);
      scwDiag.trainC(train);

      for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
        assertEquals(dpp.getPair().longValue(), scwDiag.classify(dpp.getDataPoint()).mostLikely());
      }
    }
  }

  @Test
  public void testTrainC_Full() {
    System.out.println("TrainC_Full");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final SCW.Mode mode : SCW.Mode.values()) {
      final SCW scwFull = new SCW(0.9, mode, false);
      scwFull.trainC(train);

      for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
        assertEquals(dpp.getPair().longValue(), scwFull.classify(dpp.getDataPoint()).mostLikely());
      }
    }
  }
}
