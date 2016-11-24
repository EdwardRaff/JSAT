package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.*;
import jsat.linear.*;
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
     * Paper Zou, H.,&amp;Hastie, T. (2005). Regularization and variable selection
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
        
        Random rand  = new XORWOW(45678);
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
        assertTrue(Math.abs((w.get(0)+w.get(1)*2+w.get(2))/3) < 0.4);
        
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
    
    private static class DumbWeightHolder implements Classifier, SimpleWeightVectorModel
    {
        public Vec w;
        public double b;

        @Override
        public CategoricalResults classify(DataPoint data)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void trainC(ClassificationDataSet dataSet)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public boolean supportsWeightedData()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Classifier clone()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Vec getRawWeight(int index)
        {
            return w;
        }

        @Override
        public double getBias(int index)
        {
            return b;
        }

        @Override
        public int numWeightsVecs()
        {
            return 1;
        }
        
    }
    
    @Test
    public void testWarmOther()
    {
        //Had difficulty making a problem hard enough to show improvment in warms tart but also fast to run. 
        //so made test check that warm from some weird value gets to the same places
        Random rand  = new XORWOW(5679);
        ClassificationDataSet train = new ClassificationDataSet(600, new CategoricalData[0], new CategoricalData(2));
        
        for(int i = 0; i < 200; i++)
        {
            double Z1 = rand.nextDouble()*20-10;
            double Z2 = rand.nextDouble()*20-10;
            
            Vec v = new DenseVector(train.getNumNumericalVars());
            for(int j = 0; j < v.length(); j++)
            {
                if (j > 500)
                {
                    if (j % 2 == 0)
                        v.set(j, Z2 * ((j + 1) / 600.0) + rand.nextGaussian() / (j + 1));
                    else
                        v.set(j, Z1 * ((j + 1) / 600.0) + rand.nextGaussian() / (j + 1));
                }
                else
                    v.set(j, rand.nextGaussian()*20);
            }
            
            train.addDataPoint(v, (int) (Math.signum(Z1+0.1*Z2)+1)/2);
        }
        
        NewGLMNET truth = new NewGLMNET(0.001);
        truth.setTolerance(1e-11);
        truth.trainC(train);
        
        
        DumbWeightHolder dumb = new DumbWeightHolder();
        dumb.w = DenseVector.random(train.getNumNumericalVars()).normalized();
        dumb.b = rand.nextDouble();
        
        
        
        NewGLMNET warm = new NewGLMNET(0.001);
        warm.setTolerance(1e-7);
        warm.trainC(train, dumb);
        
        assertEquals(0, warm.getRawWeight().subtract(truth.getRawWeight()).pNorm(2), 1e-4);
        assertEquals(0, warm.getBias()-truth.getBias(), 1e-4);
    }

    
}
