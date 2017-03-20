
package jsat.classifiers.linear.kernelized;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
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
public class ForgetronTest
{   
    public ForgetronTest()
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
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        for(boolean selfTuned : new boolean[]{true, false})
        {
            ClassificationDataSet train = FixedProblems.getCircles(1000, 0.0, RandomUtil.getRandom(), 1.0, 4.0);
            
            Forgetron instance = new Forgetron(new RBFKernel(0.5), 40);
            instance.setSelfTurned(selfTuned);
            instance.setEpochs(30);
            
            //add some miss labled data to get the error code to cick in and get exercised
            for(int i = 0; i < 500; i+=20)
            {
                DataPoint dp = train.getDataPoint(i);
                int y = train.getDataPointCategory(i);
                int badY = (y == 0) ? 1 : 0;
                train.addDataPoint(dp, badY);
            }

            ClassificationDataSet test = FixedProblems.getCircles(100, 0.0, RandomUtil.getRandom(), 1, 4);

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.3);//given some leway due to label noise
        }
        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        for(boolean selfTuned : new boolean[]{true, false})
        {
            ClassificationDataSet train = FixedProblems.getCircles(1000, 0.0, RandomUtil.getRandom(), 1.0, 4.0);
            
            Forgetron instance = new Forgetron(new RBFKernel(0.5), 40);
            instance.setSelfTurned(selfTuned);
            instance.setEpochs(30);
            
            //add some miss labled data to get the error code to cick in and get exercised
            for(int i = 0; i < 500; i+=20)
            {
                DataPoint dp = train.getDataPoint(i);
                int y = train.getDataPointCategory(i);
                int badY = (y == 0) ? 1 : 0;
                train.addDataPoint(dp, badY);
            }

            ClassificationDataSet test = FixedProblems.getCircles(100, 0.0, RandomUtil.getRandom(), 1, 4);
            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.3);//given some leway due to label noise
        }
    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        Forgetron instance = new Forgetron(new RBFKernel(0.5), 100);
        instance.setEpochs(30);
        
        ClassificationDataSet t1 = FixedProblems.getCircles(500, 0.0, RandomUtil.getRandom(), 1, 4);
        ClassificationDataSet t2 = FixedProblems.getCircles(500, 0.0, RandomUtil.getRandom(), 2.0, 10.0);

        instance = instance.clone();

        instance.trainC(t1);

        Forgetron result = instance.clone();
        
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }
}
