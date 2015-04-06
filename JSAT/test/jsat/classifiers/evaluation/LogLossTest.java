package jsat.classifiers.evaluation;

import jsat.classifiers.evaluation.LogLoss;
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
public class LogLossTest
{
    
    public LogLossTest()
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
     * Test of getScore method, of class LogLoss.
     */
    @Test
    public void testGetScore()
    {
        System.out.println("getScore");
        LogLoss scorer = new LogLoss();
        LogLoss otherHalf = scorer.clone();
        
        assertEquals(scorer, otherHalf);
        assertEquals(scorer.hashCode(), otherHalf.hashCode());
        assertTrue(otherHalf.lowerIsBetter());
        
        assertFalse(scorer.equals(""));
        assertFalse(scorer.hashCode() == "".hashCode());
        
        scorer.prepare(new CategoricalData(4));
        otherHalf.prepare(new CategoricalData(4));
        //from "On Using and Computing the Kappa Statistic"
        //correct
        scorer.addResult(new CategoricalResults(new double[]{0.9, 0.1, 0.0, 0.0}), 0, 317.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.0, 0.8, 0.2, 0.0}), 1, 120.0);
        scorer.addResult(new CategoricalResults(new double[]{0.0, 0.0, 0.9, 0.0}), 2, 60.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.0, 0.0, 0.0, 1.0}), 3, 8.0);
        //wrong
        scorer.addResult(new CategoricalResults(new double[]{0.1, 0.9, 0.0, 0.0}), 0, 23.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.0, 0.2, 0.9, 0.0}), 1, 61.0);
        
        scorer.addResults(otherHalf);

        double loss = 317*Math.log(0.9);
        loss += 120*Math.log(0.8);
        loss +=  60*Math.log(0.9);
        loss +=   8*Math.log(1.0);
        
        loss +=  23*Math.log(0.1);
        loss +=  61*Math.log(0.2);
        
        assertEquals(-loss/(317+120+60+8+23+61), scorer.getScore(), 1e-3);
        assertEquals(-loss/(317+120+60+8+23+61), scorer.clone().getScore(), 1e-3);
    }

}
