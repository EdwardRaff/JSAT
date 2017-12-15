package jsat.regression;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
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
public class RidgeRegressionTest
{
    public RidgeRegressionTest()
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
    public void testTrain_RegressionDataSet_Executor()
    {
        System.out.println("train");
        Random rand = new Random(2);
        
        for(RidgeRegression.SolverMode mode : RidgeRegression.SolverMode.values())
        {
            RidgeRegression regressor = new RidgeRegression(1e-9, mode);

            regressor.train(FixedProblems.getLinearRegression(400, rand), true);

            for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, new Random(3)).getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = regressor.regress(dpp.getDataPoint());
                
                double relErr = (truth-pred)/truth;
                
                assertEquals(0.0, relErr, 0.05);
            }
        }
    }
    
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(2);
        
        for(RidgeRegression.SolverMode mode : RidgeRegression.SolverMode.values())
        {
            RidgeRegression regressor = new RidgeRegression(1e-9, mode);

            regressor.train(FixedProblems.getLinearRegression(400, rand));

            for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, new Random(3)).getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = regressor.regress(dpp.getDataPoint());
                
                double relErr = (truth-pred)/truth;
                
                assertEquals(0.0, relErr, 0.05);
            }
        }
    }
}
