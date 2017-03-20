
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
public class PassiveAggressiveTest
{
    
    public PassiveAggressiveTest()
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
     * Test of trainC method, of class PassiveAggressive.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(400, RandomUtil.getRandom());
        
        for(PassiveAggressive.Mode mode : PassiveAggressive.Mode.values())
        {
            PassiveAggressive pa = new PassiveAggressive();
            pa.setMode(mode);
            pa.trainC(train);

            ClassificationDataSet test = FixedProblems.get2ClassLinear(400, RandomUtil.getRandom());

            for(DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), pa.classify(dpp.getDataPoint()).mostLikely());
        }
    }

    /**
     * Test of train method, of class PassiveAggressive.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(123);
        
        for(PassiveAggressive.Mode mode : PassiveAggressive.Mode.values())
        {
            PassiveAggressive pa = new PassiveAggressive();
            pa.setMode(mode);
            pa.setEps(0.00001);
            pa.setEpochs(10);
            pa.setC(20);
            pa.train(FixedProblems.getLinearRegression(400, rand));

            for(DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = pa.regress(dpp.getDataPoint());

                double relErr = (truth-pred)/truth;
                assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
            }
        }
    }

}
