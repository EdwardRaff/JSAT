package jsat.datatransform.featureselection;

import java.util.*;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import jsat.utils.IntSet;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class SBSTest {

  public SBSTest() {
  }

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
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
    Random rand = new Random(13);
    int t0 = 1, t1 = 5, t2 = 8;

    ClassificationDataSet cds = SFSTest.
            generate3DimIn10(rand, t0, t1, t2);

    SBS sbs = new SBS.SBSFactory(1e-3, (Classifier) new NearestNeighbour(7), 1, 7).clone().getTransform(cds).clone();
    Set<Integer> found = sbs.getSelectedNumerical();

    Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));
    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(sbs);
    assertEquals(3, cds.getNumFeatures());
  }

  @Test
  public void testTransformR() {
    System.out.println("transformR");
    Random rand = new Random(13);
    int t0 = 1, t1 = 5, t2 = 8;

    RegressionDataSet cds = SFSTest.
            generate3DimIn10R(rand, t0, t1, t2);

    SBS sbs = new SBS.SBSFactory(1.0, new MultipleLinearRegression(), 1, 7).clone().getTransform(cds).clone();
    Set<Integer> found = sbs.getSelectedNumerical();

    Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));
    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(sbs);
    assertEquals(3, cds.getNumFeatures());
  }
}
