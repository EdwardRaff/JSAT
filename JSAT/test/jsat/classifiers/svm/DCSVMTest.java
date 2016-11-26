package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
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
public class DCSVMTest
{
    static private ExecutorService ex;
    
    public DCSVMTest()
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
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(600, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            DCSVM classifier = new DCSVM(new RBFKernel(0.5));
            classifier.setCacheMode(cacheMode);
            classifier.setC(10);
            classifier.setClusterSampleSize(200);//make smaller to test sub-sampling
            classifier.trainC(trainSet, ex);

            for (int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }

        trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            DCSVM classifier = new DCSVM(new RBFKernel(0.5));
            classifier.setCacheMode(cacheMode);
            classifier.setC(10);
            classifier.setEndLevel(0);
            classifier.trainC(trainSet, ex);
            

            for (int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(600, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            DCSVM classifier = new DCSVM(new RBFKernel(0.5));
            classifier.setCacheMode(cacheMode);
            classifier.setC(10);
            classifier.setClusterSampleSize(200);//make smaller to test sub-sampling
            classifier.trainC(trainSet);

            for (int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }

        trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            DCSVM classifier = new DCSVM(new RBFKernel(0.5));
            classifier.setCacheMode(cacheMode);
            classifier.setC(10);
            classifier.setEndLevel(0);
            classifier.trainC(trainSet);
            

            for (int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }


}
