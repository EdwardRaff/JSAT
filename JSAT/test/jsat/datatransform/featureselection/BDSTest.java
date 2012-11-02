
package jsat.datatransform.featureselection;

import java.util.*;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class BDSTest
{
    
    public BDSTest()
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
     * Test of transform method, of class BDS.
     */
    @Test
    public void testTransformC()
    {
        System.out.println("transformC");
        Random rand = new Random(13);
        int t0 = 1, t1 = 5, t2 = 8;
        Set<Integer> shouldHave = new HashSet<Integer>();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        
        ClassificationDataSet cds = SFSTest.
                generate3DimIn10(rand, t0, t1, t2);
        
        BDS bds = new BDS(3, cds, new NearestNeighbour(7), 5);
        Set<Integer> found = bds.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
    }
    
    @Test
    public void testTransformR()
    {
        System.out.println("transformR");
        Random rand = new Random(13);
        int t0 = 1, t1 = 5, t2 = 8;
        Set<Integer> shouldHave = new HashSet<Integer>();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        
        RegressionDataSet rds = SFSTest.
                generate3DimIn10R(rand, t0, t1, t2);
        
        BDS bds = new BDS(3, rds, new MultipleLinearRegression(), 5);
        Set<Integer> found = bds.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
    }

}
