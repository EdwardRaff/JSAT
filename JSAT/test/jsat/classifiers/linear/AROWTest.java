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
public class AROWTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public AROWTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of classify method, of class AROW.
   */
  @Test
  public void testTrain_C() {
    System.out.println("train_C");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

    final AROW arow0 = new AROW(1, true);
    final AROW arow1 = new AROW(1, false);

    arow0.trainC(train);
    arow1.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), arow0.classify(dpp.getDataPoint()).mostLikely());
    }
    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), arow1.classify(dpp.getDataPoint()).mostLikely());
    }
  }

}
