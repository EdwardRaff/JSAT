package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
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
public class SMIDASTest
{
    
    public SMIDASTest()
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
     * Test of trainC method, of class SMIDAS.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        final ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());
        
        final SMIDAS smidas = new SMIDAS(0.1);
        smidas.setLoss(StochasticSTLinearL1.Loss.LOG);
        smidas.trainC(train);
        
        final ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());
        
        for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
          assertEquals(dpp.getPair().longValue(), smidas.classify(dpp.getDataPoint()).mostLikely());
        }
        
    }

    /**
     * Test of train method, of class SMIDAS.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        final Random rand = new Random(123);
        
        final SMIDAS smidas = new SMIDAS(0.02);
        smidas.setMinScaled(-1);
        smidas.setLoss(StochasticSTLinearL1.Loss.SQUARED);
        smidas.train(FixedProblems.getLinearRegression(500, rand));
        
        for(final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            final double truth = dpp.getPair();
            final double pred = smidas.regress(dpp.getDataPoint());
            
            final double relErr = (truth-pred)/truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
}
