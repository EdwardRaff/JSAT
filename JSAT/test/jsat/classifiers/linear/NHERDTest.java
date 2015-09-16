package jsat.classifiers.linear;

import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;

/**
 *
 * @author Edward Raff
 */
public class NHERDTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public NHERDTest() {
  }

  @Test
  public void testTrain_C() {
    System.out.println("train_C");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random(132));

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random(231));

    for (final NHERD.CovMode mode : NHERD.CovMode.values()) {
      final NHERD nherd0 = new NHERD(1, mode);
      nherd0.trainC(train);

      for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
        assertEquals(dpp.getPair().longValue(), nherd0.classify(dpp.getDataPoint()).mostLikely());
      }
    }

  }

}
