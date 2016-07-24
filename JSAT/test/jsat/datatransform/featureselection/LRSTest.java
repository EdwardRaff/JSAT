
package jsat.datatransform.featureselection;

import java.util.*;

import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
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
        Random rand = new XORWOW(13);
        int t0 = 1, t1 = 5, t2 = 8;
        Set<Integer> shouldHave = new IntSet();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        
        ClassificationDataSet cds = SFSTest.
                generate3DimIn10(rand, t0, t1, t2);
        //L > R
        LRS lrs = new LRS(6, 3, (Classifier) new NearestNeighbour(3), 5).clone();
        lrs.fit(cds);
        Set<Integer> found = lrs.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        ClassificationDataSet copyData = cds.getTwiceShallowClone();
        copyData.applyTransform(lrs);
        assertEquals(shouldHave.size(), copyData.getNumFeatures());
        
        //L < R (Leave 1 left then add 2 back
        lrs = new LRS( 2, 10-1, (Classifier)new NearestNeighbour(3),5).clone();
        lrs.fit(cds);
        found = lrs.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        cds.applyTransform(lrs);
        assertEquals(shouldHave.size(), cds.getNumFeatures());
    }
    
    @Test
    public void testTransformR()
    {
        System.out.println("transformR");
        Random rand = new XORWOW(13);
        int t0 = 1, t1 = 5, t2 = 8;
        Set<Integer> shouldHave = new IntSet();
        shouldHave.addAll(Arrays.asList(t0, t1, t2));
        
        RegressionDataSet cds = SFSTest.
                generate3DimIn10R(rand, t0, t1, t2);
        //L > R
        LRS lrs = new LRS(6, 3, (Classifier) new NearestNeighbour(3), 5).clone();
        lrs.fit(cds);
        Set<Integer> found = lrs.getSelectedNumerical();
        
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        RegressionDataSet copyData = cds.getTwiceShallowClone();
        copyData.applyTransform(lrs);
        assertEquals(shouldHave.size(), copyData.getNumFeatures());
        
        //L < R (Leave 1 left then add 2 back
        lrs = new LRS( 2, 10-1, (Classifier)new NearestNeighbour(3),5).clone();
        lrs.fit(cds);
        found = lrs.getSelectedNumerical();
        
        assertEquals(shouldHave.size(), found.size());
        assertTrue(shouldHave.containsAll(found));
        cds.applyTransform(lrs);
        assertEquals(shouldHave.size(), cds.getNumFeatures());
    }


}
