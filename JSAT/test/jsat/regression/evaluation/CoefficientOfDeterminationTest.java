package jsat.regression.evaluation;

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
public class CoefficientOfDeterminationTest
{
    
    public CoefficientOfDeterminationTest()
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
     * Test of getScore method, of class CoefficientOfDetermination.
     */
    @Test
    public void testGetScore()
    {
        System.out.println("getScore");
        CoefficientOfDetermination scorer = new CoefficientOfDetermination();
        
        CoefficientOfDetermination otherHalf = scorer.clone();
        
        assertEquals(scorer, otherHalf);
        assertEquals(scorer.hashCode(), otherHalf.hashCode());
        assertTrue(scorer.lowerIsBetter());
        
        assertFalse(scorer.equals(""));
        assertFalse(scorer.hashCode() == "".hashCode());
        
        double[] pred = new double[]
        {
            0, 2, 4, 6, 8, 9
        };
        
        double[] truth = new double[]
        {
            0.5, 2, 3, 1, 8.5, 10
        };
        
        scorer.prepare();
        otherHalf.prepare();
        
        for(int i = 0; i < pred.length/2; i++)
            scorer.addResult(pred[i], truth[i], 1);
        for(int i = pred.length/2; i < pred.length; i++)
            otherHalf.addResult(pred[i], truth[i], 1);
        
        scorer.addResults(otherHalf);
        
        assertEquals(0.69894, scorer.getScore(), 1e-1);
        
        scorer = scorer.clone();
        
        assertEquals(0.69894, scorer.getScore(), 1e-1);
        
        
    }
    
}
