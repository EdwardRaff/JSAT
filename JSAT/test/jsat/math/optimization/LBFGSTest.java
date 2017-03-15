package jsat.math.optimization;

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.FunctionVec;
import jsat.utils.random.RandomUtil;
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
public class LBFGSTest
{
    
    public LBFGSTest()
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
     * Test of optimize method, of class LBFGS.
     */
    @Test
    public void testOptimize()
    {
        System.out.println("optimize");
        Random rand = RandomUtil.getRandom();
        Vec x0 = new DenseVector(3);//D=3 means one local minima for easy evaluation
        for(int i = 0; i < x0.length(); i++)
            x0.set(i, rand.nextDouble()+0.5);//make sure we get to the right local optima

        RosenbrockFunction f = new RosenbrockFunction();
        FunctionVec fp = f.getDerivative();
        LBFGS instance = new LBFGS();
        
        for(LineSearch lineSearch : new LineSearch[]{new BacktrackingArmijoLineSearch(), new WolfeNWLineSearch()})
        {
            instance.setLineSearch(lineSearch);
            Vec w = new DenseVector(x0.length());
            instance.optimize(1e-5, w, x0, f, fp, null);

            for(int i = 0; i <w.length(); i++)
                assertEquals(1.0, w.get(i), 1e-3);
            assertEquals(0.0, f.f(w), 1e-4);
        }
    }
}
