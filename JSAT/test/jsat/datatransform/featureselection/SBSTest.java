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
public class SBSTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public SBSTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testTransformC() {
    System.out.println("transformC");
    final Random rand = new Random(13);
    final int t0 = 1, t1 = 5, t2 = 8;

    final ClassificationDataSet cds = SFSTest.generate3DimIn10(rand, t0, t1, t2);

    final SBS sbs = new SBS.SBSFactory(1e-3, (Classifier) new NearestNeighbour(7), 1, 7).clone().getTransform(cds)
        .clone();
    final Set<Integer> found = sbs.getSelectedNumerical();

    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));
    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(sbs);
    assertEquals(3, cds.getNumFeatures());
  }

  @Test
  public void testTransformR() {
    System.out.println("transformR");
    final Random rand = new Random(13);
    final int t0 = 1, t1 = 5, t2 = 8;

    final RegressionDataSet cds = SFSTest.generate3DimIn10R(rand, t0, t1, t2);

    final SBS sbs = new SBS.SBSFactory(1.0, new MultipleLinearRegression(), 1, 7).clone().getTransform(cds).clone();
    final Set<Integer> found = sbs.getSelectedNumerical();

    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));
    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(sbs);
    assertEquals(3, cds.getNumFeatures());
  }
}
