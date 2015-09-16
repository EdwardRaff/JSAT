package jsat.datatransform.featureselection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Random;
import java.util.Set;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.classifiers.ClassificationDataSet;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class ReliefFTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public ReliefFTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of transform method, of class ReliefF.
   */
  @Test
  public void testTransformC() {
    System.out.println("transformC");
    final Random rand = new Random(13);
    final int t0 = 1, t1 = 5, t2 = 8;
    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));

    final ClassificationDataSet cds = SFSTest.generate3DimIn10(rand, t0, t1, t2);

    final ReliefF relieff = new ReliefF.ReliefFFactory(3, 50, 7, new EuclideanDistance()).clone().getTransform(cds)
        .clone();
    final Set<Integer> found = new IntSet(relieff.getKeptNumeric());

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(relieff);
    assertEquals(shouldHave.size(), cds.getNumFeatures());
  }

}
