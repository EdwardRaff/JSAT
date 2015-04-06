package jsat.classifiers.evaluation;

import jsat.classifiers.evaluation.AUC;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
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
public class AUCTest
{
    
    public AUCTest()
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
     * Test of getScore method, of class AUC.
     */
    @Test
    public void testGetScore()
    {
        System.out.println("getScore");
        AUC scorer = new AUC();
        AUC otherHalf = scorer.clone();
        
        assertEquals(scorer, otherHalf);
        assertEquals(scorer.hashCode(), otherHalf.hashCode());
        assertFalse(scorer.lowerIsBetter());
        
        assertFalse(scorer.equals(""));
        assertFalse(scorer.hashCode() == "".hashCode());
        
        scorer.prepare(new CategoricalData(2));
        otherHalf.prepare(new CategoricalData(2));
        
        scorer.addResult(new CategoricalResults(new double[]{0.2, 0.8}), 1, 3.0);
        scorer.addResult(new CategoricalResults(new double[]{0.4, 0.6}), 0, 2.0);
        scorer.addResult(new CategoricalResults(new double[]{0.6, 0.4}), 1, 1.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.7, 0.3}), 0, 1.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.9, 0.1}), 1, 1.0);
        otherHalf.addResult(new CategoricalResults(new double[]{1.0, 0.0}), 0, 1.0);
        
        scorer.addResults(otherHalf);
        
        double P = 2.0+1.0+1.0;
        double N = 3.0+1.0+1.0;
        //AUC dosn't make as much sense with so few data points...
        assertEquals((3+2)/(P*N), scorer.getScore(), 1e-1);
        assertEquals((3+2)/(P*N), scorer.clone().getScore(), 1e-1);
        
        
        
        
    }

}
