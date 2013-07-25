package jsat.classifiers.linear.kernelized;

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
public class ProjectronTest
{
    static private ExecutorService ex;
    
    public ProjectronTest()
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
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(550, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));


        Projectron classifierPP = new Projectron(new RBFKernel(0.5), 0.1, true);
        classifierPP.trainC(trainSet, ex);
        
        Projectron classifier = new Projectron(new RBFKernel(0.5), 0.1, false);
        classifier.trainC(trainSet, ex);

        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifierPP.classify(testSet.getDataPoint(i)).mostLikely());
        
        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());

    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(550, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));


        Projectron classifierPP = new Projectron(new RBFKernel(0.5), 0.1, true);
        classifierPP.trainC(trainSet);
        
        Projectron classifier = new Projectron(new RBFKernel(0.5), 0.1, false);
        classifier.trainC(trainSet);

        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifierPP.classify(testSet.getDataPoint(i)).mostLikely());
        
        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());

    }
}
