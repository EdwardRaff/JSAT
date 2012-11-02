
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
public class LRSTest
{
    
    public LRSTest()
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
     * Test of transform method, of class LRS.
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
        //L > R
        LRS lrs = new LRS(6, 3, cds, new NearestNeighbour(3), 5);
        Set<Integer> found = lrs.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        
        //L < R (Leave 1 left then add 2 back
        lrs = new LRS(2, 10-1, cds, new NearestNeighbour(3), 5);
        found = lrs.getSelectedNumerical();
        
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
        
        RegressionDataSet cds = SFSTest.
                generate3DimIn10R(rand, t0, t1, t2);
        //L > R
        LRS lrs = new LRS(6, 3, cds, new MultipleLinearRegression(), 5);
        Set<Integer> found = lrs.getSelectedNumerical();
        
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        
        //L < R (Leave 1 left then add 2 back
        lrs = new LRS(2, 10-1, cds, new MultipleLinearRegression(), 5);
        found = lrs.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
    }


}
