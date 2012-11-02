
package jsat.datatransform.featureselection;

import java.util.*;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.*;

/**
 *
 * @author Edward Raff
 */
public class SFSTest
{
    
    public SFSTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
    }
    
    @AfterClass
    public static void tearDownClass()
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
     * Test of transform method, of class SequentialForwardSelection.
     */
    @Test
    public void testTransform()
    {
        System.out.println("transform");
        Random rand = new Random(12343);
        int t0 = 1, t1 = 5, t2 = 8;
        
        
        ClassificationDataSet cds = generate3DimIn10(rand, t0, t1, t2);
       
        SFS sfs = new SFS(3, 7, cds, 
                new NearestNeighbour(7), 5, 1e-3);
        Set<Integer> found = sfs.getSelectedNumerical();
        
        Set<Integer> shouldHave = new HashSet<Integer>();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
    }
    
    @Test
    public void testTransformR()
    {
        System.out.println("transformR");
        Random rand = new Random(12343);
        int t0 = 1, t1 = 5, t2 = 8;
        
        
        RegressionDataSet rds = generate3DimIn10R(rand, t0, t1, t2);
       
        SFS sfs = new SFS(3, 7, rds, 
                new MultipleLinearRegression(), 5, 10);
        Set<Integer> found = sfs.getSelectedNumerical();
        
        Set<Integer> shouldHave = new HashSet<Integer>();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
    }

    /**
     * Creates a naive test case where 4 classes that can be separated with 3 
     * features are placed into a 10 dimensional space. The other 7 dimensions
     * are all small random noise values. 
     * 
     * @param rand source of randomness
     * @param t0 the true index in the 10 dimensional space to place the first value
     * @param t1 the true index in the 10 dimensional space to place the second value
     * @param t2 the true index in the 10 dimensional space to place the third value
     */
    public static ClassificationDataSet generate3DimIn10(Random rand, 
            int t0, int t1, int t2)
    {
        ClassificationDataSet cds = new ClassificationDataSet(10, 
                new CategoricalData[0], new CategoricalData(4));
        int cSize = 40;
        for(int i = 0; i < cSize; i++)
        {
            Vec dv = DenseVector.random(10, rand);
            dv.mutableDivide(3);
            
            dv.set(t0, 5.0);
            dv.set(t1, 5.0);
            dv.set(t2, 0.0);
            cds.addDataPoint(dv, new int[0], 0);
            
        }
        
        for(int i = 0; i < cSize; i++)
        {
            Vec dv = DenseVector.random(10, rand);
            dv.mutableDivide(3);
            
            dv.set(t0, 5.0);
            dv.set(t1, 5.0);
            dv.set(t2, 5.0);
            cds.addDataPoint(dv, new int[0], 1);
        }
        
        for(int i = 0; i < cSize; i++)
        {
            Vec dv = DenseVector.random(10, rand);
            dv.mutableDivide(3);
            
            dv.set(t0, 5.0);
            dv.set(t1, 0.0);
            dv.set(t2, 5.0);
            cds.addDataPoint(dv, new int[0], 2);
        }
        
        for(int i = 0; i < cSize; i++)
        {
            Vec dv = DenseVector.random(10, rand);
            dv.mutableDivide(3);
            
            dv.set(t0, 0.0);
            dv.set(t1, 5.0);
            dv.set(t2, 5.0);
            cds.addDataPoint(dv, new int[0], 3);
        }
        
        return cds;
    }
    
    public static RegressionDataSet generate3DimIn10R(Random rand, 
            int t0, int t1, int t2)
    {
        RegressionDataSet cds = new RegressionDataSet(10, new CategoricalData[0]);
        int cSize = 40;
        for(int i = 0; i < cSize; i++)
        {
            Vec dv = DenseVector.random(10, rand);
            
            cds.addDataPoint(dv, new int[0], dv.get(t0)*6 + dv.get(t1)*4 + dv.get(t2)*8);
            
        }
        
        return cds;
    }

}
