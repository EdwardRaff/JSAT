
package jsat.distributions.multivariate;

import java.util.List;
import java.util.Random;
import jsat.linear.DenseVector;
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
public class DirichletTest
{
    
    public DirichletTest()
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
     * Test of logPdf method, of class Dirichlet.
     */
    @Test
    public void testLogPdf()
    {
        System.out.println("logPdf");
        assertEquals(Math.log(18.0/25.0), new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(3.0/10.0, 2.0/10.0, 5.0/10.0)), 1e-13);
        assertEquals(Math.log(42.0/25.0), new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(7.0/10.0, 2.0/10.0, 1.0/10.0)), 1e-13);
        assertEquals(Math.log(96.0/25.0), new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        
        //If it dosnt sum to 1, its not a possible value 
        assertEquals(-Double.MAX_VALUE, new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(5.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(-Double.MAX_VALUE, new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(1.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(-Double.MAX_VALUE, new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(-4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(-Double.MAX_VALUE, new Dirichlet(toDenseVec(2, 3, 1)).logPdf(toDenseVec(-4.0/10.0, 4.0/10.0, 10.0/10.0)), 1e-13);
    }

    /**
     * Test of pdf method, of class Dirichlet.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        
        assertEquals(18.0/25.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(3.0/10.0, 2.0/10.0, 5.0/10.0)), 1e-13);
        assertEquals(42.0/25.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(7.0/10.0, 2.0/10.0, 1.0/10.0)), 1e-13);
        assertEquals(96.0/25.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        
        //If it dosnt sum to 1, its not a possible value 
        assertEquals(0.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(5.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(0.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(1.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(0.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(-4.0/10.0, 4.0/10.0, 2.0/10.0)), 1e-13);
        assertEquals(0.0, new Dirichlet(toDenseVec(2, 3, 1)).pdf(toDenseVec(-4.0/10.0, 4.0/10.0, 10.0/10.0)), 1e-13);
    }

    /**
     * Test of setUsingData method, of class Dirichlet.
     */
    @Test
    public void testSetUsingData()
    {
        System.out.println("setUsingData");
        List<Vec> dataSet = null;
        Dirichlet instance = new Dirichlet(DenseVector.toDenseVec(2.5, 2.5, 2.5));
        dataSet = instance.sample(500, new Random(1));
        
        Dirichlet setI = new Dirichlet(DenseVector.toDenseVec(2.0, 2.0, 2.0));
        setI.setUsingData(dataSet);
        
        //5% error would be fine 
        Vec alpha = setI.getAlphas();
        for(int i = 0; i < alpha.length(); i++ )
            assertEquals(0, Math.abs(alpha.get(i)-2.5)/2.5, 0.05);
    }

}
