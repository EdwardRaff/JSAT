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
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.linear.Vec;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class SFSTest {

  /**
   * Creates a naive test case where 4 classes that can be separated with 3
   * features are placed into a 10 dimensional space. The other 7 dimensions are
   * all small random noise values.
   *
   * @param rand
   *          source of randomness
   * @param t0
   *          the true index in the 10 dimensional space to place the first
   *          value
   * @param t1
   *          the true index in the 10 dimensional space to place the second
   *          value
   * @param t2
   *          the true index in the 10 dimensional space to place the third
   *          value
   */
  public static ClassificationDataSet generate3DimIn10(final Random rand, final int t0, final int t1, final int t2) {
    final ClassificationDataSet cds = new ClassificationDataSet(10, new CategoricalData[0], new CategoricalData(4));
    final int cSize = 40;
    for (int i = 0; i < cSize; i++) {
      final Vec dv = Vec.random(10, rand);
      dv.mutableDivide(3);

      dv.set(t0, 5.0);
      dv.set(t1, 5.0);
      dv.set(t2, 0.0);
      cds.addDataPoint(dv, new int[0], 0);

    }

    for (int i = 0; i < cSize; i++) {
      final Vec dv = Vec.random(10, rand);
      dv.mutableDivide(3);

      dv.set(t0, 5.0);
      dv.set(t1, 5.0);
      dv.set(t2, 5.0);
      cds.addDataPoint(dv, new int[0], 1);
    }

    for (int i = 0; i < cSize; i++) {
      final Vec dv = Vec.random(10, rand);
      dv.mutableDivide(3);

      dv.set(t0, 5.0);
      dv.set(t1, 0.0);
      dv.set(t2, 5.0);
      cds.addDataPoint(dv, new int[0], 2);
    }

    for (int i = 0; i < cSize; i++) {
      final Vec dv = Vec.random(10, rand);
      dv.mutableDivide(3);

      dv.set(t0, 0.0);
      dv.set(t1, 5.0);
      dv.set(t2, 5.0);
      cds.addDataPoint(dv, new int[0], 3);
    }

    return cds;
  }

  public static RegressionDataSet generate3DimIn10R(final Random rand, final int t0, final int t1, final int t2) {
    final RegressionDataSet cds = new RegressionDataSet(10, new CategoricalData[0]);
    final int cSize = 40;
    for (int i = 0; i < cSize; i++) {
      final Vec dv = Vec.random(10, rand);

      cds.addDataPoint(dv, new int[0], dv.get(t0) * 6 + dv.get(t1) * 4 + dv.get(t2) * 8);

    }

    return cds;
  }

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public SFSTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of transform method, of class SequentialForwardSelection.
   */
  @Test
  public void testTransform() {
    System.out.println("transform");
    final Random rand = new Random(12343);
    final int t0 = 1, t1 = 5, t2 = 8;

    final ClassificationDataSet cds = generate3DimIn10(rand, t0, t1, t2);

    final SFS sfs = new SFS.SFSFactory(1e-3, (Classifier) new NearestNeighbour(7), 3, 7).clone().getTransform(cds)
        .clone();
    final Set<Integer> found = sfs.getSelectedNumerical();

    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));
    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    cds.applyTransform(sfs);
    assertEquals(3, cds.getNumFeatures());
  }

  @Test
  public void testTransformR() {
    System.out.println("transformR");
    final Random rand = new Random(12343);
    final int t0 = 1, t1 = 5, t2 = 8;

    final RegressionDataSet rds = generate3DimIn10R(rand, t0, t1, t2);

    final SFS sfs = new SFS.SFSFactory(10, new MultipleLinearRegression(), 3, 7).clone().getTransform(rds).clone();
    final Set<Integer> found = sfs.getSelectedNumerical();

    final Set<Integer> shouldHave = new IntSet();
    shouldHave.addAll(Arrays.asList(t0, t1, t2));
    assertEquals(shouldHave.size(), found.size());
    assertTrue(shouldHave.containsAll(found));
    rds.applyTransform(sfs);
    assertEquals(3, rds.getNumFeatures());
  }

}
