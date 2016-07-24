
package jsat.datatransform.featureselection;

import java.util.*;

import jsat.classifiers.ClassificationDataSet;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.IntSet;
import jsat.utils.random.XORWOW;

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
public class ReliefFTest
{
    
    public ReliefFTest()
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
     * Test of transform method, of class ReliefF.
     */
    @Test
    public void testTransformC()
    {
        System.out.println("transformC");
        Random rand = new XORWOW(13);
        int t0 = 1, t1 = 5, t2 = 8;
        Set<Integer> shouldHave = new IntSet();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        
        ClassificationDataSet cds = SFSTest.
                generate3DimIn10(rand, t0, t1, t2);
        
        ReliefF relieff = new ReliefF(3, 50, 7, new EuclideanDistance()).clone();
        relieff.fit(cds);
        Set<Integer> found = new IntSet(relieff.getKeptNumeric());
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        cds.applyTransform(relieff);
        assertEquals(shouldHave.size(), cds.getNumFeatures());
    }

}
