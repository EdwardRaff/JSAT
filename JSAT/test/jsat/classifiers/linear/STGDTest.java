package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.*;
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
public class STGDTest
{
    
    public STGDTest()
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
     * Test of train method, of class STGD.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(123);
        
        STGD scd = new STGD(5, 0.1, Double.POSITIVE_INFINITY, 0.1);
        scd.train(FixedProblems.getLinearRegression(400, rand));
        
        for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(400, rand).getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = scd.regress(dpp.getDataPoint());
            
            double relErr = (truth-pred)/truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }


    /**
     * Test of trainC method, of class STGD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(400, RandomUtil.getRandom());
        
        STGD scd = new STGD(5, 0.5, Double.POSITIVE_INFINITY, 0.1);
        scd.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(400, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), scd.classify(dpp.getDataPoint()).mostLikely());
    }
}
