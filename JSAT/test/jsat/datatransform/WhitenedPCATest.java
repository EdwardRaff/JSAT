/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.datatransform;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class WhitenedPCATest {

  @BeforeClass
  public static void setUpClass() throws Exception {
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
  }

  public WhitenedPCATest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of setUpTransform method, of class WhitenedPCA.
   */
  @Test
  public void testTransform() {
    System.out.println("testTransform");
    final NormalM normal = new NormalM(new DenseVector(3), new DenseMatrix(
        new double[][] { { 133.138, -57.278, 40.250 }, { -57.278, 25.056, -17.500 }, { 40.250, -17.500, 12.250 }, }));

    final List<Vec> sample = normal.sample(500, new Random(17));
    final List<DataPoint> dataPoints = new ArrayList<DataPoint>(sample.size());
    for (final Vec v : sample) {
      dataPoints.add(new DataPoint(v, new int[0], new CategoricalData[0]));
    }

    final SimpleDataSet data = new SimpleDataSet(dataPoints);

    final DataTransform transform = new WhitenedPCA(data, 0.0);

    data.applyTransform(transform);

    final Matrix whiteCov = MatrixStatistics.covarianceMatrix(MatrixStatistics.meanVector(data), data);

    assertTrue(Matrix.eye(3).equals(whiteCov, 1e-8));
  }

}
