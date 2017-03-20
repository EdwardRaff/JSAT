package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import static java.lang.Math.*;
import java.util.Arrays;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author Edward Raff
 */
public class SVMnoBiasTest
{
    static private ExecutorService ex;
    
    public SVMnoBiasTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        ex.shutdown();
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
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

            for(SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
            {
                SVMnoBias classifier = new SVMnoBias(new RBFKernel(0.5));
                classifier.setCacheMode(cacheMode);
                classifier.setC(10);
                classifier.trainC(trainSet, ex);
                
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
            }
    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));


            for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
            {
                SVMnoBias classifier = new SVMnoBias(new RBFKernel(0.5));
                classifier.setCacheMode(cacheMode);
                classifier.setC(10);
                classifier.trainC(trainSet);
                
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
                
                //test warm start off corrupted solution
                double[] a = classifier.alphas;
                Random rand = RandomUtil.getRandom();
                for(int i = 0; i < a.length; i++)
                    a[i] = min(max(a[i]+rand.nextDouble()*2-1, 0), 10);
                
                SVMnoBias classifier2 = new SVMnoBias(new RBFKernel(0.5));
                classifier2.setCacheMode(cacheMode);
                classifier2.setC(10);
                classifier2.trainC(trainSet, a);
                
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    assertEquals(testSet.getDataPointCategory(i), classifier2.classify(testSet.getDataPoint(i)).mostLikely());
            }
    }


}
