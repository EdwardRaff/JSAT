/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.math.rootfinding;

import jsat.linear.Vec;
import jsat.math.rootfinding.RiddersMethod;
import jsat.math.Function;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class RiddersMethodTest
{
    /**
     * Root at 0
     */
    Function sinF = new Function() {

        /**
		 * 
		 */
		private static final long serialVersionUID = -4942395915907632276L;

		public double f(double... x)
        {
            return sin(x[0]);
        }
        
        public double f(Vec x)
        {
            return f(x.arrayCopy());
        }
    };
    
    /**
     * Root at 0 + 2nd param
     */
    Function sinFp1 = new Function() {

        /**
		 * 
		 */
		private static final long serialVersionUID = -6913574202545691152L;

		public double f(double... x)
        {
            return sin(x[0]+x[1]);
        }
        
        public double f(Vec x)
        {
            return f(x.arrayCopy());
        }
    };
    
    /**
     * Root at approx -4.87906
     */
    Function polyF = new Function() {

        /**
		 * 
		 */
		private static final long serialVersionUID = -206733171455524905L;

		public double f(double... x)
        {
            double xp = x[0];
            
            return pow(xp, 3)+5*pow(xp,2)+xp+2;
        }
        
        public double f(Vec x)
        {
            return f(x.arrayCopy());
        }
    };
    
    public RiddersMethodTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
    }

    /**
     * Test of root method, of class RiddersMethod.
     */
    @Test
    public void testRoot_4args()
    {
        System.out.println("root");
        double eps = 1e-15;
        double result = RiddersMethod.root(-PI/2, PI/2, sinF);
        assertEquals(0, result, eps);
        
        result = RiddersMethod.root(-6, 6, polyF);
        assertEquals(-4.8790576334840479813, result, eps);
        
        result = RiddersMethod.root(-6, 6, polyF, 0);
        assertEquals(-4.8790576334840479813, result, eps);
        
        
        result = RiddersMethod.root(-PI / 2, PI / 2, sinFp1, 0.0, 1.0);
        assertEquals(-1.0, result, eps);
        
        try
        {
            result = RiddersMethod.root(-PI / 2, PI / 2, sinFp1);
            fail("Should not have run");
        }
        catch (Exception ex)
        {
        }
    }

    /**
     * Test of root method, of class RiddersMethod.
     */
    @Test
    public void testRoot_5args()
    {
        System.out.println("root");
        double eps = 1e-15;
        double result = RiddersMethod.root(eps, -PI/2, PI/2, sinF);
        assertEquals(0, result, eps);
        
        result = RiddersMethod.root(eps, -6, 6, polyF);
        assertEquals(-4.8790576334840479813, result, eps);
        
        result = RiddersMethod.root(eps, -6, 6, 0, polyF);
        assertEquals(-4.8790576334840479813, result, eps);
        
        
        result = RiddersMethod.root(eps, -PI / 2, PI / 2, sinFp1, 0.0, 1.0);
        assertEquals(-1.0, result, eps);
        
        try
        {
            result = RiddersMethod.root(eps, -PI / 2, PI / 2, sinFp1);
            fail("Should not have run");
        }
        catch (Exception ex)
        {
        }
    }
    
    @Test
    public void testRoot_6args()
    {
        System.out.println("root");
        double eps = 1e-15;
        double result = RiddersMethod.root(eps, -PI/2, PI/2, 0, sinF);
        assertEquals(0, result, eps);
        
        result = RiddersMethod.root(eps, -PI/2, PI/2, 0, sinFp1, 0.0, 1.0);
        assertEquals(-1.0, result, eps);
        
        result = RiddersMethod.root(eps, -PI/2, PI/2, 1, sinFp1, 3.0, 0.0);
        assertEquals(PI-3.0, result, eps);
        
        result = RiddersMethod.root(eps, -6, 6, 0, polyF);
        assertEquals(-4.8790576334840479813, result, eps);
    }

    /**
     * Test of root method, of class RiddersMethod.
     */
    @Test
    public void testRoot_7args()
    {
        System.out.println("root");
        double eps = 1e-13;
        int maxIterations = 1000;
        double result = RiddersMethod.root(eps, maxIterations, -PI/2, PI/2, 0, sinF);
        assertEquals(0, result, eps);
        
        result = RiddersMethod.root(eps, maxIterations, -PI/2, PI/2, 0, sinFp1, 0.0, 1.0);
        assertEquals(-1.0, result, eps);
        
        result = RiddersMethod.root(eps, maxIterations, -PI/2, PI/2, 1, sinFp1, 3.0, 0.0);
        assertEquals(PI-3.0, result, eps);
        
        result = RiddersMethod.root(eps, maxIterations, -6, 6, 0, polyF);
        assertEquals(-4.8790576334840479813, result, eps);
    }
}
