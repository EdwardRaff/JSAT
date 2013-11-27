
package jsat.math;

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
public class MathTricksTest
{
    
    public MathTricksTest()
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
     * Test of hornerPolyR method, of class MathTricks.
     */
    @Test
    public void testHornerPolyR()
    {
        System.out.println("hornerPolyR");
        double[] coef = new double[]{1, -4, 0, 8, 1};
        assertEquals(1, MathTricks.hornerPolyR(coef, 0), 1e-15);
        assertEquals(6, MathTricks.hornerPolyR(coef, 1), 1e-15);
        assertEquals(481, MathTricks.hornerPolyR(coef, 6), 1e-15);
        assertEquals(2113, MathTricks.hornerPolyR(coef, -6), 1e-15);
        
    }

    /**
     * Test of hornerPoly method, of class MathTricks.
     */
    @Test
    public void testHornerPoly()
    {
        System.out.println("hornerPoly");
        double[] coef = new double[]{1, 8, 0, -4, 1};
        assertEquals(1, MathTricks.hornerPoly(coef, 0), 1e-15);
        assertEquals(6, MathTricks.hornerPoly(coef, 1), 1e-15);
        assertEquals(481, MathTricks.hornerPoly(coef, 6), 1e-15);
        assertEquals(2113, MathTricks.hornerPoly(coef, -6), 1e-15);
        
    }
}
