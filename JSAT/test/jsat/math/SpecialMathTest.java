
package jsat.math;

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
public class SpecialMathTest
{
    
    public SpecialMathTest()
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
     * Test of digamma method, of class SpecialMath.
     */
    @Test
    public void testZeta()
    {
        System.out.println("digamma");
        
        //Values in this range have crappy accuracy... sad panda
        
        double[] inputNW = new double[]
        {
            -10.5, -2, -1.5,
        };
        
        double[] expectedNW = new double[]
        {
            0.01114612247394282,0.,-0.02548520188983304,
        };
        
        for(int i = 0; i < inputNW.length; i++)
        {
            assertEquals(expectedNW[i], SpecialMath.zeta(inputNW[i]), 1e-01);
        }
        
        //Decent
        double[] input = new double[]
        {
            -0.5, 0.2, 0.5, 0.9, 1.1, 1.3, 2,
        };
        double[] expected = new double[]
        {
            -0.2078862249773546,-0.7339209248963376,-1.460354508809586,
            -9.43011401940225,10.5844484649508,3.931949211809544,
            1.644934066848226
        };
        
        
        for(int i = 0; i < input.length; i++)
        {
            assertEquals(expected[i], SpecialMath.zeta(input[i]), 1e-5);
        }
        
        
        //Very good
        double[] inputVG = new double[]
        {
             2.6, 10.4, 15, 20, 40.0, 60
        };
        double[] expectedVG = new double[]
        {
            1.305477809072781,1.000751620674465,1.000030588236307,
            1.000000953962033,1.00000000000091,0.999999999999997
        };
        for(int i = 0; i < expectedVG.length; i++)
        {
            assertEquals(expectedVG[i], SpecialMath.zeta(inputVG[i]), 1e-14);
        }
    }
    
    /**
     * Test of digamma method, of class SpecialMath.
     */
    @Test
    public void testDigamma()
    {
        System.out.println("digamma");
        double[] input = new double[]
        {
            -77.5, -1.5, 1, 1.4, 2, 5, 6, 9, 20, 100
        };
        double[] expected = new double[]
        {
            4.356715675057194,0.7031566406452434,-0.5772156649015328,
            -0.0613845445851161,0.4227843350984672,1.506117668431801,
            1.7061176684318,2.14064147795561,2.97052399224215,4.600161852738087
        };
        for(int i = 0; i < input.length; i++)
            assertEquals(expected[i], SpecialMath.digamma(input[i]), 1e-14);
    }

    /**
     * Test of reLnBn method, of class SpecialMath.
     */
    @Test
    public void testReLnBn()
    {
        System.out.println("reLnBn");
        int[] input = new int[]
        {
            0, 1, 2, 3, 4, 5, 6, 10, 40, 50, 70, 100
        };
        
        double[] expected = new double[]
        {
            0.,-0.6931471805599453,-1.791759469228055,Double.NEGATIVE_INFINITY,
            -3.401197381662155,Double.NEGATIVE_INFINITY,-3.737669618283368,
            -2.580216829592325,37.49870423894444,57.2770608118657,
            102.4807960976827,180.6448160951889
        };
        
        for(int i = 0; i < input.length; i++)
            assertEquals(expected[i], SpecialMath.reLnBn(input[i]), 1e-11);
    }

    /**
     * Test of bernoulli method, of class SpecialMath.
     */
    @Test
    public void testBernoulli()
    {
        System.out.println("bernoulli");
        int[] input = new int[]
        {
            0, 1, 2, 3, 4, 5, 6, 10, 14, 20
        };
        
        double[] expected = new double[]
        {
            1.,-0.5,0.1666666666666667,0.,-0.03333333333333333,0.,
            0.02380952380952381,0.07575757575757576,1.166666666666667,
            -529.1242424242425
        };
        
        for(int i = 0; i < input.length; i++)
            assertEquals(expected[i], SpecialMath.bernoulli(input[i]), 1e-11);
    }


}
