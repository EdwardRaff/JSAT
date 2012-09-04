/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.datatransform;

import java.util.*;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.*;
import static org.junit.Assert.assertTrue;
import org.junit.*;

/**
 *
 * @author Edward Raff
 */
public class WhitenedZCATest
{
    
    public WhitenedZCATest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of setUpTransform method, of class WhitenedZCA.
     */
    @Test
    public void testTransform()
    {
        System.out.println("testTransform");
        
        NormalM normal = new NormalM(new DenseVector(3), new DenseMatrix(new double[][]
        {
            {10.3333,   -4.1667,    3.0000},
            {-4.1667,    2.3333,   -1.5000},
            { 3.0000,   -1.5000,    1.0000},
        }));
        
        List<Vec> sample = normal.sample(500, new Random(17));
        List<DataPoint> dataPoints  = new ArrayList<DataPoint>(sample.size());
        for( Vec v : sample)
            dataPoints.add(new DataPoint(v, new int[0], new CategoricalData[0]));
        
        SimpleDataSet data = new SimpleDataSet(dataPoints);
        
        DataTransform transform = new WhitenedZCA(data, 0);
        
        data.applyTransform(transform);
        
        Matrix whiteCov = MatrixStatistics.covarianceMatrix(MatrixStatistics.meanVector(data), data);
        
        assertTrue(Matrix.eye(3).equals(whiteCov, 1e-8));

    }
}
