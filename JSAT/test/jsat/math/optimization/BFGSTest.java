package jsat.math.optimization;

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.FunctionVec;
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
public class BFGSTest
{
    
    public BFGSTest()
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
     * Test of optimize method, of class BFGS.
     */
    @Test
    public void testOptimize()
    {
        System.out.println("optimize");
        final Random rand = new Random();
        final Vec x0 = new DenseVector(20);
        for(int i = 0; i < x0.length(); i++) {
          x0.set(i, rand.nextDouble());
        }

        final RosenbrockFunction f = new RosenbrockFunction();
        final FunctionVec fp = f.getDerivative();
        final BFGS instance = new BFGS();
        
        for(final LineSearch lineSearch : new LineSearch[]{new BacktrackingArmijoLineSearch(), new WolfeNWLineSearch()})
        {
            instance.setLineSearch(lineSearch);
            final Vec w = new DenseVector(x0.length());
            instance.optimize(1e-4, w, x0, f, fp, null);

            for(int i = 0; i <w.length(); i++) {
              assertEquals(1.0, w.get(i), 1e-4);
            }
            assertEquals(0.0, f.f(w), 1e-4);
        }
    }
}
