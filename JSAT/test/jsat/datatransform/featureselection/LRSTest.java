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
import jsat.classifiers.Classifier;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class LRSTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public LRSTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of transform method, of class LRS.
   */
  @Test
  public void testTransformC() {
    System.out.println("transformC");
    final Random rand = new Random(13);
    final int t0 = 1, t1 = 5, t2 = 8;
    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));

    final ClassificationDataSet cds = SFSTest.generate3DimIn10(rand, t0, t1, t2);
    // L > R
    LRS lrs = new LRS.LRSFactory((Classifier) new NearestNeighbour(3), 6, 3).clone().getTransform(cds).clone();
    Set<Integer> found = lrs.getSelectedNumerical();

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    final ClassificationDataSet copyData = cds.getTwiceShallowClone();
    copyData.applyTransform(lrs);
    assertEquals(shouldHave.size(), copyData.getNumFeatures());

    // L < R (Leave 1 left then add 2 back
    lrs = new LRS.LRSFactory((Classifier) new NearestNeighbour(3), 2, 10 - 1).clone().getTransform(cds).clone();
    found = lrs.getSelectedNumerical();

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(lrs);
    assertEquals(shouldHave.size(), cds.getNumFeatures());
  }

  @Test
  public void testTransformR() {
    System.out.println("transformR");
    final Random rand = new Random(13);
    final int t0 = 1, t1 = 5, t2 = 8;
    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));

    final RegressionDataSet cds = SFSTest.generate3DimIn10R(rand, t0, t1, t2);
    // L > R
    LRS lrs = new LRS.LRSFactory(new MultipleLinearRegression(), 6, 3).clone().getTransform(cds).clone();
    Set<Integer> found = lrs.getSelectedNumerical();

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    final RegressionDataSet copyData = cds.getTwiceShallowClone();
    copyData.applyTransform(lrs);
    assertEquals(shouldHave.size(), copyData.getNumFeatures());

    // L < R (Leave 1 left then add 2 back
    lrs = new LRS.LRSFactory(new MultipleLinearRegression(), 2, 10 - 1).clone().getTransform(cds).clone();
    found = lrs.getSelectedNumerical();

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(lrs);
    assertEquals(shouldHave.size(), cds.getNumFeatures());
  }

}
