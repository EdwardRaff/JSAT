package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.SquaredLoss;
import jsat.regression.RegressionDataSet;
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
public class LinearBatchTest
{
    static ExecutorService ex;
    
    public LinearBatchTest()
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
    public void testClassifyBinary()
    {
        System.out.println("binary classifiation");
        
        LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());
        
        linearBatch.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyBinaryMT()
    {
        System.out.println("binary classifiation MT");
        
        LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());
        
        linearBatch.trainC(train, ex);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyMulti()
    {
        System.out.println("multi class classification");
        
        LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());
        
        linearBatch.trainC(train);
        
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyMultiMT()
    {
        System.out.println("multi class classification MT");
        
        LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());
        
        linearBatch.trainC(train, ex);
        
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("regression");
        
        LinearBatch linearBatch = new LinearBatch(new SquaredLoss(), 1e-4);
        RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());
        
        linearBatch.train(train);
        
        RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = linearBatch.regress(dpp.getDataPoint());
            double relErr = (truth-pred)/truth;
            assertEquals(0, relErr, 0.1);
        }
    }
    
    @Test
    public void testRegressionMT()
    {
        System.out.println("regression MT");
        
        LinearBatch linearBatch = new LinearBatch(new SquaredLoss(), 1e-4);
        RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());
        
        linearBatch.train(train, ex);
        
        RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = linearBatch.regress(dpp.getDataPoint());
            double relErr = (truth-pred)/truth;
            assertEquals(0, relErr, 0.01);
        }
    }

}
