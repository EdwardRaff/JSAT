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
public class BDSTest {

  public BDSTest() {
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

  /**
   * Test of transform method, of class BDS.
   */
  @Test
  public void testTransformC() {
    System.out.println("transformC");
    Random rand = new Random(13);
    int t0 = 1, t1 = 5, t2 = 8;
    Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));

    ClassificationDataSet cds = SFSTest.
            generate3DimIn10(rand, t0, t1, t2);

    BDS bds = new BDS.BDSFactory((Classifier) new NearestNeighbour(7), 3).clone().getTransform(cds).clone();
    Set<Integer> found = bds.getSelectedNumerical();

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(bds);
    assertEquals(shouldHave.size(), cds.getNumFeatures());
  }

  @Test
  public void testTransformR() {
    System.out.println("transformR");
    Random rand = new Random(13);
    int t0 = 1, t1 = 5, t2 = 8;
    Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));

    RegressionDataSet rds = SFSTest.
            generate3DimIn10R(rand, t0, t1, t2);

    BDS bds = new BDS.BDSFactory(new MultipleLinearRegression(), 3).clone().getTransform(rds).clone();
    Set<Integer> found = bds.getSelectedNumerical();

    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    rds.applyTransform(bds);
    assertEquals(shouldHave.size(), rds.getNumFeatures());
  }

}
