/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear.distancemetrics;

import jsat.linear.DenseVector;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
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
public class PearsonDistanceTest
{
    Vec x1, x2, x3, x4, x5;
    
    public PearsonDistanceTest()
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
        x1 = new DenseVector(new double[]{43, 21, 25, 42, 57, 59});
        x2 = new DenseVector(new double[]{99, 65, 79, 75, 87, 81});
        x3 = new DenseVector(new double[]{ 0, 12,  0, 38, 19,  0});
        x4 = new DenseVector(new double[]{ 0, 60, 27,  0, 13,  9});
        x5 = new DenseVector(new double[]{ 0,  0, 25, 42,  0,  0});
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of correlation method, of class PearsonDistance.
     */
    @Test
    public void testCorrelation()
    {
        System.out.println("correlation");
        
        Vec[][] pairs = new Vec[][]
        {
            {x1, x2},
            {x2, x1},
            {x2, x2},
            {x3, x4},
            {x4, x3},
            {x1, x5},
            {x3, x5},
            {x4, x5},
        };
        
        double[] expectedVals = new double[]
        {
            0.5298089018901744,
            0.5298089018901744,
            1.0, 
            -0.16532953228838151,
            -0.16532953228838151,
            -0.2587220950032312,
            0.5995147431422027,
            -0.2625496015918526,
        };
        
        for(int i = 0; i < pairs.length; i++)
        {
            double result = PearsonDistance.correlation(pairs[i][0], pairs[i][1], false);
            double expected = expectedVals[i];
            assertEquals(expected, result, 1e-14);
            result = PearsonDistance.correlation(pairs[i][1], pairs[i][0], false);
            assertEquals(expected, result, 1e-14);
        }
        //on sparse inputs
        for(int i = 0; i < pairs.length; i++)
        {
            double result = PearsonDistance.correlation(new SparseVector(pairs[i][0]), new SparseVector(pairs[i][1]), false);
            double expected = expectedVals[i];
            assertEquals(expected, result, 1e-14);
            result = PearsonDistance.correlation(new SparseVector(pairs[i][1]), new SparseVector(pairs[i][0]), false);
            assertEquals(expected, result, 1e-14);
        }
                
    }
    
    @Test
    public void testCorrelationBNZ()
    {
        System.out.println("correlation");
        
        Vec[][] pairs = new Vec[][]
        {
            {x1, x2},
            {x2, x1},
            {x2, x2},
            {x3, x4},
            {x4, x3},
            {x1, x5},
            {x3, x5},
            {x4, x5},
        };
        
        double[] expectedVals = new double[]
        {
            0.5298089018901744,
            0.5298089018901744,
            1.0, 
            -0.7254024447917231,
            -0.7254024447917231,
            0.7425697235686509,
            1.0,
            1.0,
        };
        
        for(int i = 0; i < pairs.length; i++)
        {
            double result = PearsonDistance.correlation(pairs[i][0], pairs[i][1], true);
            double expected = expectedVals[i];
            assertEquals(expected, result, 1e-14);
            result = PearsonDistance.correlation(pairs[i][1], pairs[i][0], true);
            assertEquals(expected, result, 1e-14);
        }
        //on sparse inputs
        for(int i = 0; i < pairs.length; i++)
        {
            double result = PearsonDistance.correlation(new SparseVector(pairs[i][0]), new SparseVector(pairs[i][1]), true);
            double expected = expectedVals[i];
            assertEquals(expected, result, 1e-14);
            result = PearsonDistance.correlation(new SparseVector(pairs[i][1]), new SparseVector(pairs[i][0]), true);
            assertEquals(expected, result, 1e-14);
        }
                
    }
}
