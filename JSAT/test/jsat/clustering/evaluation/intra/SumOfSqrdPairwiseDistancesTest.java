package jsat.clustering.evaluation.intra;

import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.distancemetrics.MinkowskiDistance;

/**
 *
 * @author Edward Raff
 */
public class SumOfSqrdPairwiseDistancesTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public SumOfSqrdPairwiseDistancesTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of evaluate method, of class SumOfSqrdPairwiseDistances.
   */
  @Test
  public void testEvaluate_3args() {
    System.out.println("evaluate");
    final int[] designations = new int[10];
    final SimpleDataSet dataSet = new SimpleDataSet(new CategoricalData[0], 1);
    final int clusterID = 2;
    for (int i = 0; i < 10; i++) {
      dataSet.add(new DataPoint(new DenseVector(new double[] { i })));
    }
    designations[1] = designations[3] = designations[5] = designations[9] = clusterID;

    SumOfSqrdPairwiseDistances instance = new SumOfSqrdPairwiseDistances();
    final double expResult = 280 / (2 * 4);
    double result = instance.evaluate(designations, dataSet, clusterID);
    assertEquals(expResult, result, 1e-14);

    // minkowski p=2 is equivalent to euclidean, but implementation wont check
    // for that
    // just to make sure in future, make it not quite 2 - but numericaly close
    // enought
    instance = new SumOfSqrdPairwiseDistances(new MinkowskiDistance(Math.nextUp(2)));

    result = instance.evaluate(designations, dataSet, clusterID);
    assertEquals(expResult, result, 1e-14);

  }

  /**
   * Test of evaluate method, of class SumOfSqrdPairwiseDistances.
   */
  @Test
  public void testEvaluate_List() {
    System.out.println("evaluate");
    final SimpleDataSet dataSet = new SimpleDataSet(new CategoricalData[0], 1);
    for (int i = 0; i < 10; i++) {
      dataSet.add(new DataPoint(new DenseVector(new double[] { i })));
    }
    final List<DataPoint> dataPoints = dataSet.getBackingList();
    SumOfSqrdPairwiseDistances instance = new SumOfSqrdPairwiseDistances();
    final double expResult = 1650.0 / (2 * 10);
    double result = instance.evaluate(dataPoints);
    assertEquals(expResult, result, 1e-14);

    instance = new SumOfSqrdPairwiseDistances(new MinkowskiDistance(Math.nextUp(2)));

    result = instance.evaluate(dataPoints);
    assertEquals(expResult, result, 1e-14);
  }

}
