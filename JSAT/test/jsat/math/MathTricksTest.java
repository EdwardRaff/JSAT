
package jsat.math;

import java.util.Arrays;
import jsat.linear.DenseVector;
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
     * Test of logSumExp method, of class MathTricks.
     */
    @Test
    public void testLogSumExp_Vec_double()
    {
        System.out.println("logSumExp");
        Vec vals = DenseVector.toDenseVec(10, 10, 3, -20, 1);
        double maxValue = 10.0;
        double expResult = 10.693664692512399350;
        double result = MathTricks.logSumExp(vals, maxValue);
        assertEquals(expResult, result, 1e-15);
    }

    /**
     * Test of logSumExp method, of class MathTricks.
     */
    @Test
    public void testLogSumExp_doubleArr_double()
    {
        System.out.println("logSumExp");
        double[] vals = new double[] {10, 10, 3, -20, 1};
        double maxValue = 10.0;
        double expResult = 10.693664692512399350;
        double result = MathTricks.logSumExp(vals, maxValue);
        assertEquals(expResult, result, 1e-15);
    }

    /**
     * Test of softmax method, of class MathTricks.
     */
    @Test
    public void testSoftmax_doubleArr_boolean()
    {
        System.out.println("softmax");
        double[] x = new double[]{3, 1, 2};
        double[] x2 = Arrays.copyOf(x, x.length);
        
        double[] xExpected = new double[]{0.6652409557748218895, 0.090030573170380457998, 0.24472847105479765};
        double[] x2Expected = new double[]{0.64391425988797231, 0.087144318742032567489, 0.23688281808991013};
        
        MathTricks.softmax(x, false);
        for(int i = 0; i < x.length; i++)
            assertEquals(xExpected[i], x[i], 1e-15);
        MathTricks.softmax(x2, true);
        for(int i = 0; i < x2.length; i++)
            assertEquals(x2Expected[i], x2[i], 1e-15);
    }

    /**
     * Test of softmax method, of class MathTricks.
     */
    @Test
    public void testSoftmax_Vec_boolean()
    {
        System.out.println("softmax");
        Vec x = new DenseVector(new double[]{3, 1, 2});
        Vec x2 = x.clone();
        
        Vec xExpected = new DenseVector(new double[]{0.6652409557748218895, 0.090030573170380457998, 0.24472847105479765});
        Vec x2Expected = new DenseVector(new double[]{0.64391425988797231, 0.087144318742032567489, 0.23688281808991013});
        
        MathTricks.softmax(x, false);
        for(int i = 0; i < x.length(); i++)
            assertEquals(xExpected.get(i), x.get(i), 1e-15);
        MathTricks.softmax(x2, true);
        for(int i = 0; i < x2.length(); i++)
            assertEquals(x2Expected.get(i), x2.get(i), 1e-15);
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
