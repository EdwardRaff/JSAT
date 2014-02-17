package jsat.classifiers.linear.kernelized;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.distributions.kernels.RBFKernel;
import jsat.lossfunctions.HingeLoss;
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
public class BOGDTest
{
    
    public BOGDTest()
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

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        BOGD bogd = new BOGD(new RBFKernel(0.5), 30, 0.5, 1e-3, 10, new HingeLoss());
        bogd.setUniformSampling(false);
        bogd.trainC(trainSet);
        
        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), bogd.classify(testSet.getDataPoint(i)).mostLikely());
        
        bogd.setUniformSampling(true);
        bogd.trainC(trainSet);
        
        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), bogd.classify(testSet.getDataPoint(i)).mostLikely());

    }
}
