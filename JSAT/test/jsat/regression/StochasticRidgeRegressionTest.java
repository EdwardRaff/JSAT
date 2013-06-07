package jsat.regression;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.math.decayrates.DecayRate;
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
public class StochasticRidgeRegressionTest
{
    static ExecutorService ex;
    
    public StochasticRidgeRegressionTest()
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
    public void testTrain_RegressionDataSet_Executor()
    {
        System.out.println("train");
        Random rand = new Random(2);
        
        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression regressor = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.1);

            regressor.train(FixedProblems.getLinearRegression(400, rand), ex);

            for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, new Random(3)).getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = regressor.regress(dpp.getDataPoint());
                
                double relErr = (truth-pred)/truth;
                
                assertEquals(0.0, relErr, 0.10);//extra wiggle room due to stochastic on a small problem
            }
        }
    }
    
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(2);
        
        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression regressor = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.1);

            regressor.train(FixedProblems.getLinearRegression(400, rand));

            for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, new Random(3)).getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = regressor.regress(dpp.getDataPoint());
                
                double relErr = (truth-pred)/truth;
                
                assertEquals(0.0, relErr, 0.10);//extra wiggle room due to stochastic on a small problem
            }
        }
    }
}
