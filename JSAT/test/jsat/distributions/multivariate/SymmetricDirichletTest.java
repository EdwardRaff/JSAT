
package jsat.distributions.multivariate;

import java.util.List;
import java.util.Random;
import jsat.linear.Vec;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import static jsat.linear.DenseVector.*;

/**
 *
 * @author Edward Raff
 */
public class SymmetricDirichletTest
{
    
    public SymmetricDirichletTest()
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
     * Test of clone method, of class SymmetricDirichlet.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        SymmetricDirichlet first = new SymmetricDirichlet(2.5, 3);
        SymmetricDirichlet clone = first.clone();
        
        first.setAlpha(1.5);
        
        assertEquals(2.5, clone.getAlpha(), 0);
        assertEquals(1.5, first.getAlpha(), 0);
    }

    /**
     * Test of logPdf method, of class SymmetricDirichlet.
     */
    @Test
    public void testLogPdf()
    {
        System.out.println("logPdf");
        assertEquals(Math.log(18.0/5.0),        new SymmetricDirichlet(2.0, 3).logPdf(toDenseVec(3.0/10.0, 2.0/10.0, 5.0/10.0)), 1e-13);
        assertEquals(Math.log(2.0),             new SymmetricDirichlet(1.0, 3).logPdf(toDenseVec(7.0/10.0, 2.0/10.0, 1.0/10.0)), 1e-13);
        assertEquals(Math.log(16128.0/3125.0),  new SymmetricDirichlet(3.0, 3).logPdf(toDenseVec(4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        
        //If it dosnt sum to 1, its not a possible value 
        assertEquals(-Double.MAX_VALUE, new SymmetricDirichlet(2.0, 3).logPdf(toDenseVec(5.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(-Double.MAX_VALUE, new SymmetricDirichlet(2.0, 3).logPdf(toDenseVec(1.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(-Double.MAX_VALUE, new SymmetricDirichlet(2.0, 3).logPdf(toDenseVec(-4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(-Double.MAX_VALUE, new SymmetricDirichlet(2.0, 3).logPdf(toDenseVec(-4.0/10.0, 4.0/10.0, 10.0/10.0)), 1e-13);
    }

    /**
     * Test of pdf method, of class SymmetricDirichlet.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        assertEquals(18.0/5.0,       new SymmetricDirichlet(2.0, 3).pdf(toDenseVec(3.0/10.0, 2.0/10.0, 5.0/10.0)), 1e-13);
        assertEquals(2.0,            new SymmetricDirichlet(1.0, 3).pdf(toDenseVec(7.0/10.0, 2.0/10.0, 1.0/10.0)), 1e-13);
        assertEquals(16128.0/3125.0, new SymmetricDirichlet(3.0, 3).pdf(toDenseVec(4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        
        //If it dosnt sum to 1, its not a possible value 
        assertEquals(0.0, new SymmetricDirichlet(2.0, 3).pdf(toDenseVec(5.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(0.0, new SymmetricDirichlet(2.0, 3).pdf(toDenseVec(1.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(0.0, new SymmetricDirichlet(2.0, 3).pdf(toDenseVec(-4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(0.0, new SymmetricDirichlet(2.0, 3).pdf(toDenseVec(-4.0/10.0, 4.0/10.0, 10.0/10.0)), 1e-13);
    }

    /**
     * Test of setUsingData method, of class SymmetricDirichlet.
     */
    @Test
    public void testSetUsingData()
    {
        System.out.println("setUsingData");
        List<Vec> dataSet = null;
        SymmetricDirichlet instance = new SymmetricDirichlet(2.5, 3);
        dataSet = instance.sample(500, new Random(1));
        
        SymmetricDirichlet setI = new SymmetricDirichlet(1.0, 3);
        setI.setUsingData(dataSet);
        
        //We will be happy with %5 accuracy
        assertEquals(0, (setI.getAlpha()-2.5)/2.5, 0.05);
    }
}
