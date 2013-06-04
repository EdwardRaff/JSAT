/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear.kernelized;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.svm.SupportVectorLearner;
import jsat.distributions.kernels.RBFKernel;
import jsat.parameters.Parameter;
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
public class CSKLRBatchTest
{
    static private ExecutorService ex;
    
    public CSKLRBatchTest()
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
     * Test of trainC method, of class CSKLRBatch.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));
        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values())
        {
            CSKLRBatch csklr = new CSKLRBatch(0.5, new RBFKernel(0.5), 10, mode, SupportVectorLearner.CacheMode.NONE);
            csklr.trainC(trainSet, ex);
            
            for(int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), csklr.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }

    /**
     * Test of trainC method, of class CSKLRBatch.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));
        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values())
        {
            CSKLRBatch csklr = new CSKLRBatch(0.5, new RBFKernel(0.5), 10, mode, SupportVectorLearner.CacheMode.NONE);
            csklr.trainC(trainSet);
            
            for(int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), csklr.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }

}
