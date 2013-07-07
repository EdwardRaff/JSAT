package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
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
public class DCDsTest
{
    static private ExecutorService threadPool;
    
    public DCDsTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        threadPool.shutdown();
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
     * Test of trainC method, of class DCDs.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());
        
        DCDs instance = new DCDs();
        instance.trainC(train, threadPool);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }

    /**
     * Test of trainC method, of class DCDs.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

        DCDs instance = new DCDs();
        instance.trainC(train);

        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

        for (DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testTrain_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");
        Random rand = new Random();

        DCDs dcds = new DCDs();
        dcds.train(FixedProblems.getLinearRegression(400, rand), threadPool);

        for (DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = dcds.regress(dpp.getDataPoint());

            double relErr = (truth - pred) / truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
    
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random();

        DCDs dcds = new DCDs();
        dcds.train(FixedProblems.getLinearRegression(400, rand));

        for (DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = dcds.regress(dpp.getDataPoint());

            double relErr = (truth - pred) / truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
}
