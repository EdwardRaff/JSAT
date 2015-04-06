package jsat.classifiers.evaluation;

import jsat.classifiers.evaluation.MatthewsCorrelationCoefficient;
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
public class MatthewsCorrelationCoefficientTest
{
    
    public MatthewsCorrelationCoefficientTest()
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
     * Test of getScore method, of class MatthewsCorrelationCoefficient.
     */
    @Test
    public void testGetScore()
    {
        System.out.println("getScore");
        MatthewsCorrelationCoefficient scorer = new MatthewsCorrelationCoefficient();
        MatthewsCorrelationCoefficient otherHalf = scorer.clone();
        
        assertEquals(scorer, otherHalf);
        assertEquals(scorer.hashCode(), otherHalf.hashCode());
        assertFalse(otherHalf.lowerIsBetter());
        
        assertFalse(scorer.equals(""));
        assertFalse(scorer.hashCode() == "".hashCode());
        
        scorer.prepare(new CategoricalData(2));
        otherHalf.prepare(new CategoricalData(2));
        //correct
        scorer.addResult(new CategoricalResults(new double[]{1.0, 0.0}), 0, 1.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.2, 0.8}), 1, 3.0);
        scorer.addResult(new CategoricalResults(new double[]{7.0, 0.3}), 0, 1.0);
        //wrong
        otherHalf.addResult(new CategoricalResults(new double[]{0.6, 0.4}), 1, 1.0);
        scorer.addResult(new CategoricalResults(new double[]{0.4, 0.6}), 0, 2.0);
        otherHalf.addResult(new CategoricalResults(new double[]{0.9, 0.1}), 1, 1.0);
        
        scorer.addResults(otherHalf);
        double tp = 2, tn = 3, fp = 2, fn = 2;
        assertEquals((tp*tn-fp*fn)/Math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)), scorer.getScore(), 1e-2);
        assertEquals((tp*tn-fp*fn)/Math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)), scorer.clone().getScore(), 1e-2);
    }

}
