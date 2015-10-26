package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.utils.SystemInfo;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class DCDTest
{
    static private ExecutorService threadPool;
    
    public DCDTest()
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
    
    /**
     * Test of trainC method, of class DCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());
        
        final DCD instance = new DCD();
        instance.trainC(train, threadPool);
        
        final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
        for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
          assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
        }
    }

    /**
     * Test of trainC method, of class DCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        final ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random());

        final DCD instance = new DCD();
        instance.trainC(train);

        final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

        for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
          assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test
    public void testTrain_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");
        final Random rand = new Random();

        final DCD dcd = new DCD();
        dcd.train(FixedProblems.getLinearRegression(400, rand), threadPool);

        for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            final double truth = dpp.getPair();
            final double pred = dcd.regress(dpp.getDataPoint());

            final double relErr = (truth - pred) / truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
    
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        final Random rand = new Random();

        final DCD dcd = new DCD();
        dcd.train(FixedProblems.getLinearRegression(400, rand));

        for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            final double truth = dpp.getPair();
            final double pred = dcd.regress(dpp.getDataPoint());

            final double relErr = (truth - pred) / truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }

}
