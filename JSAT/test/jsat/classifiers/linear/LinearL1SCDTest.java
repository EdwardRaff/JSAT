
package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.FixedProblems;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.regression.RegressionDataSet;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward
 */
public class LinearL1SCDTest
{
    
    public LinearL1SCDTest()
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
     * Test of train method, of class LinearL1SCD.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(123);
        
        LinearL1SCD scd = new LinearL1SCD();
        scd.setMinScaled(-1);
        scd.setLoss(StochasticSTLinearL1.Loss.SQUARED);
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
     * Test of trainC method, of class LinearL1SCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());
        
        LinearL1SCD scd = new LinearL1SCD();
        scd.setLoss(StochasticSTLinearL1.Loss.LOG);
        scd.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), scd.classify(dpp.getDataPoint()).mostLikely());
    }
}
