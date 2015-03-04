package jsat.classifiers.linear;

import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
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
public class NewGLMNETTest
{
    /*
     * This test case is based off of the grouping example in the Elatic Net 
     * Paper Zou, H., & Hastie, T. (2005). Regularization and variable selection
     * via the elastic net. Journal of the Royal Statistical Society, Series B, 
     * 67(2), 301–320. doi:10.1111/j.1467-9868.2005.00503.x
     */
    
    public NewGLMNETTest()
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
     * Test of setC method, of class NewGLMNET.
     */
    @Test
    public void testSetC()
    {
        System.out.println("train");
        
        Random rand  = new XORWOW();
        ClassificationDataSet data = new ClassificationDataSet(6, new CategoricalData[0], new CategoricalData(2));
        
        for(int i = 0; i < 500; i++)
        {
            double Z1 = rand.nextDouble()*20-10;
            double Z2 = rand.nextDouble()*20-10;
            Vec v = DenseVector.toDenseVec(Z1, -Z1, Z1, Z2, -Z2, Z2);
            data.addDataPoint(v, (int) (Math.signum(Z1+0.1*Z2)+1)/2);
        }
        
        Vec w;
        NewGLMNET glmnet = new NewGLMNET();
        glmnet.setUseBias(false);
        
        glmnet.setC(1e-2);
        glmnet.setAlpha(1);
        
        do
        {
            glmnet.setC(glmnet.getC()-0.0001);
            glmnet.trainC(data);
            w = glmnet.getRawWeight();
        }
        while (w.nnz() > 1);//we should be able to find this pretty easily
        
        assertEquals(1, w.nnz());
        int nonZeroIndex = w.getNonZeroIterator().next().getIndex();
        assertTrue(nonZeroIndex < 3);//should be one of the more important weights
        if(nonZeroIndex == 1) //check the sign is correct
            assertEquals(-1, (int)Math.signum(w.get(nonZeroIndex)));
        else
            assertEquals(1, (int)Math.signum(w.get(nonZeroIndex)));
        
        glmnet.setC(1);
        glmnet.setAlpha(0.5);//now we should get the top 3 on
        do
        {
            glmnet.setC(glmnet.getC()*0.9);
            glmnet.trainC(data);
            w = glmnet.getRawWeight();
        }
        while (w.nnz() > 3);//we should be able to find this pretty easily
        assertEquals(3, w.nnz());
        assertEquals( 1, (int)Math.signum(w.get(0)));
        assertEquals(-1, (int)Math.signum(w.get(1)));
        assertEquals( 1, (int)Math.signum(w.get(2)));
        //also want to make sure that they are all about equal in size
        assertTrue(Math.abs((w.get(0)+w.get(1)*2+w.get(2))/3) < 0.2);
        
        glmnet.setC(1e-3);
        glmnet.setAlpha(0);//now everyone should turn on
        glmnet.trainC(data);
        w = glmnet.getRawWeight();
        assertEquals(6, w.nnz());
        assertEquals( 1, (int)Math.signum(w.get(0)));
        assertEquals(-1, (int)Math.signum(w.get(1)));
        assertEquals( 1, (int)Math.signum(w.get(2)));
        assertEquals( 1, (int)Math.signum(w.get(3)));
        assertEquals(-1, (int)Math.signum(w.get(4)));
        assertEquals( 1, (int)Math.signum(w.get(5)));
        
        
    }

    
}
