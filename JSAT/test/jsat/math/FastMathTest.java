package jsat.math;

import java.util.Collections;
import java.util.Random;
import jsat.utils.DoubleList;
import jsat.utils.random.RandomUtil;
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
public class FastMathTest
{
    Random rand;
    
    public FastMathTest()
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
        rand = RandomUtil.getRandom();
    }
    
    @After
    public void tearDown()
    {
    }
    
    public static double relErr(double expected, double actual)
    {
        return Math.abs((expected-actual)/expected);
    }

    /**
     * Test of log2 method, of class FastMath.
     */
    @Test
    public void testLog2()
    {
        System.out.println("log2");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*1000;
            relErrs.add(relErr(Math.log(x)/Math.log(2), FastMath.log2(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
    }

    /**
     * Test of log2_2pd1 method, of class FastMath.
     */
    @Test
    public void testLog2_2pd1()
    {
        System.out.println("log2_2pd1");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*1000;
            relErrs.add(relErr(Math.log(x)/Math.log(2), FastMath.log2_2pd1(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
    }

    /**
     * Test of log2_c11 method, of class FastMath.
     */
    @Test
    public void testLog2_c11()
    {
        System.out.println("log2_c11");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*1000;
            relErrs.add(relErr(Math.log(x)/Math.log(2), FastMath.log2_c11(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
    }

    /**
     * Test of pow2 method, of class FastMath.
     */
    @Test
    public void testPow2_int()
    {
        System.out.println("pow2");
        for(int i = 0; i < 20; i++)
            assertEquals(Math.pow(2,i), FastMath.pow2(i), 0.0);//docs say it must be exact
    }

    /**
     * Test of pow2 method, of class FastMath.
     */
    @Test
    public void testPow2_double()
    {
        System.out.println("pow2");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*20;
            relErrs.add(relErr(Math.pow(2, x), FastMath.pow2(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
        
    }

    /**
     * Test of pow method, of class FastMath.
     */
    @Test
    public void testPow()
    {
        System.out.println("pow");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*20;
            double y = rand.nextDouble()*20;
            relErrs.add(relErr(Math.pow(y, x), FastMath.pow(y, x)));
        }
        Collections.sort(relErrs);
        //most should have lowwer rel error for best cases
        assertTrue(relErrs.get((int) (trials*.10)) <= 1e-4);//usually does better than this
        assertTrue(relErrs.get((int) (trials*.50)) <= 1e-3);
        assertTrue(relErrs.get((int) (trials*.75)) <= 1e-2);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-1);
    }

    /**
     * Test of log method, of class FastMath.
     */
    @Test
    public void testLog()
    {
        System.out.println("log");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*20+1e-15;
            relErrs.add(relErr(Math.log(x), FastMath.log(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
    }

    /**
     * Test of exp method, of class FastMath.
     */
    @Test
    public void testExp()
    {
        System.out.println("exp");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*50-25;
            relErrs.add(relErr(Math.exp(x), FastMath.exp(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
    }

    /**
     * Test of digamma method, of class FastMath.
     */
    @Test
    public void testDigamma()
    {
        System.out.println("digamma");
        DoubleList relErrs = new DoubleList();
        int trials = 10000;
        for(int i = 0; i < trials; i++)
        {
            double x = rand.nextDouble()*50;
            relErrs.add(relErr(SpecialMath.digamma(x), FastMath.digamma(x)));
        }
        Collections.sort(relErrs);
        assertTrue(relErrs.get((int) (trials*.95)) <= 1e-3);
    }
}
