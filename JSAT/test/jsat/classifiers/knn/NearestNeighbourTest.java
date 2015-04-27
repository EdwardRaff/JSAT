/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.knn;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.distributions.Normal;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.utils.GridDataGenerator;
import jsat.utils.SystemInfo;
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
public class NearestNeighbourTest
{
    static private ClassificationDataSet easyTrain;
    static private ClassificationDataSet easyTest;
    static private ExecutorService ex;
    static private NearestNeighbour nn;
    
    public NearestNeighbourTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.05), new Random(12), 2);
        easyTrain = new ClassificationDataSet(gdg.generateData(80).getBackingList(), 0);
        easyTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
    }
    
    @Before
    public void setUp()
    {
        nn = new NearestNeighbour(1);
    }
    
    @After
    public void tearDown()
    {
    }
    
    @Test
    public void testTrainC_RegressionDataSet()
    {
        System.out.println("train");

        NearestNeighbour instance = new NearestNeighbour(7);

        RegressionDataSet train = FixedProblems.getLinearRegression(1000, new XORWOW());
        RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());

        RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
        rme.evaluateTestSet(test);

        assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.5);

    }

    @Test
    public void testTrainC_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");

        NearestNeighbour instance = new NearestNeighbour(7);

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        RegressionDataSet train = FixedProblems.getLinearRegression(1000, new XORWOW());
        RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());

        RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
        rme.evaluateTestSet(test);

        assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.5);

        ex.shutdownNow();
    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        nn.trainC(easyTrain);
        for(int i = 0; i < easyTest.getSampleSize(); i++)
            assertEquals(easyTest.getDataPointCategory(i), nn.classify(easyTest.getDataPoint(i)).mostLikely());
    }

    @Test
    public void testClone()
    {
        System.out.println("clone");
        nn.trainC(easyTrain);
        Classifier clone = nn.clone();
        for(int i = 0; i < easyTest.getSampleSize(); i++)
            assertEquals(easyTest.getDataPointCategory(i), clone.classify(easyTest.getDataPoint(i)).mostLikely());
    }

    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        nn.trainC(easyTrain, ex);
        for(int i = 0; i < easyTest.getSampleSize(); i++)
            assertEquals(easyTest.getDataPointCategory(i), nn.classify(easyTest.getDataPoint(i)).mostLikely());
    }
}
