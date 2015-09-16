/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.datatransform;

import static org.junit.Assert.assertEquals;
import java.util.Arrays;
import java.util.Set;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class RemoveAttributeTransformTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public RemoveAttributeTransformTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of clone method, of class RemoveAttributeTransform.
   */
  @Test
  public void testClone() {
    System.out.println("clone");
    final CategoricalData[] catIndo = new CategoricalData[] { new CategoricalData(2), new CategoricalData(3),
        new CategoricalData(4) };
    final int[] catVals = new int[] { 0, 1, 2 };
    final Vec numVals = DenseVector.toDenseVec(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    final DataPoint dp = new DataPoint(numVals, catVals, catIndo);
    final SimpleDataSet dataSet = new SimpleDataSet(Arrays.asList(dp));

    final Set<Integer> catToRemove = new IntSet();
    catToRemove.add(1);
    final Set<Integer> numToRemove = new IntSet();
    numToRemove.addAll(Arrays.asList(0, 2, 3));

    RemoveAttributeTransform transform = new RemoveAttributeTransform(dataSet, catToRemove, numToRemove);

    transform = transform.clone();

    final DataPoint transFormed = transform.transform(dp);
    final int[] tranCatVals = transFormed.getCategoricalValues();
    assertEquals(2, tranCatVals.length);
    assertEquals(0, tranCatVals[0]);
    assertEquals(2, tranCatVals[1]);

    final Vec tranNumVals = transFormed.getNumericalValues();
    assertEquals(4, tranNumVals.length());
    assertEquals(1.0, tranNumVals.get(0), 0.0);
    assertEquals(4.0, tranNumVals.get(1), 0.0);
    assertEquals(5.0, tranNumVals.get(2), 0.0);
    assertEquals(6.0, tranNumVals.get(3), 0.0);
  }

  /**
   * Test of consolidate method, of class RemoveAttributeTransform.
   */
  @Test
  public void testConsolidate() {
    System.out.println("consolidate");
    final CategoricalData[] catIndo = new CategoricalData[] { new CategoricalData(2), new CategoricalData(3),
        new CategoricalData(4) };
    final int[] catVals = new int[] { 0, 1, 2 };
    final Vec numVals = DenseVector.toDenseVec(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    final DataPoint dp = new DataPoint(numVals, catVals, catIndo);
    SimpleDataSet dataSet = new SimpleDataSet(Arrays.asList(dp));

    final Set<Integer> catToRemove = new IntSet();
    catToRemove.add(1);
    final Set<Integer> numToRemove = new IntSet();
    numToRemove.addAll(Arrays.asList(0, 2, 3));

    final RemoveAttributeTransform transform = new RemoveAttributeTransform(dataSet, catToRemove, numToRemove);

    DataPoint transformed = transform.transform(dp);

    catToRemove.clear();
    catToRemove.add(0);

    numToRemove.clear();
    numToRemove.addAll(Arrays.asList(0, 3));

    dataSet = new SimpleDataSet(Arrays.asList(transformed));

    final RemoveAttributeTransform transform2 = new RemoveAttributeTransform(dataSet, catToRemove, numToRemove);

    // Consolidate and make sure it is right
    transform2.consolidate(transform);
    transformed = transform2.transform(dp);

    final int[] tranCatVals = transformed.getCategoricalValues();
    assertEquals(1, tranCatVals.length);
    assertEquals(2, tranCatVals[0]);

    final Vec tranNumVals = transformed.getNumericalValues();
    assertEquals(2, tranNumVals.length());
    assertEquals(4.0, tranNumVals.get(0), 0.0);
    assertEquals(5.0, tranNumVals.get(1), 0.0);

  }

  /**
   * Test of transform method, of class RemoveAttributeTransform.
   */
  @Test
  public void testTransform() {
    System.out.println("transform");
    final CategoricalData[] catIndo = new CategoricalData[] { new CategoricalData(2), new CategoricalData(3),
        new CategoricalData(4) };
    final int[] catVals = new int[] { 0, 1, 2 };
    final Vec numVals = DenseVector.toDenseVec(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    final DataPoint dp = new DataPoint(numVals, catVals, catIndo);
    final SimpleDataSet dataSet = new SimpleDataSet(Arrays.asList(dp));

    final Set<Integer> catToRemove = new IntSet();
    catToRemove.add(1);
    final Set<Integer> numToRemove = new IntSet();
    numToRemove.addAll(Arrays.asList(0, 2, 3));

    final RemoveAttributeTransform transform = new RemoveAttributeTransform(dataSet, catToRemove, numToRemove);

    final DataPoint transFormed = transform.transform(dp);
    final int[] tranCatVals = transFormed.getCategoricalValues();
    assertEquals(2, tranCatVals.length);
    assertEquals(0, tranCatVals[0]);
    assertEquals(2, tranCatVals[1]);

    final Vec tranNumVals = transFormed.getNumericalValues();
    assertEquals(4, tranNumVals.length());
    assertEquals(1.0, tranNumVals.get(0), 0.0);
    assertEquals(4.0, tranNumVals.get(1), 0.0);
    assertEquals(5.0, tranNumVals.get(2), 0.0);
    assertEquals(6.0, tranNumVals.get(3), 0.0);
  }
}
