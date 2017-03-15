package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.SquaredLoss;
import jsat.utils.random.RandomUtil;
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
public class SCDTest
{
    
    public SCDTest()
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

    /**
     * Test of trainC method, of class SCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(400, RandomUtil.getRandom());
        
        SCD scd = new SCD(new LogisticLoss(), 1e-6, 100);
        scd.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(400, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
        {
            assertEquals(dpp.getPair().longValue(), scd.classify(dpp.getDataPoint()).mostLikely());
        }
        
    }

    /**
     * Test of train method, of class SCD.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(123);
        
        SCD scd = new SCD(new SquaredLoss(), 1e-6, 1000);//needs more iters for regression
        scd.train(FixedProblems.getLinearRegression(500, rand));
        
        for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = scd.regress(dpp.getDataPoint());
            
            double relErr = (truth-pred)/truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
}
