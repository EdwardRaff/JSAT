
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
public class ROMMATest
{
    
    public ROMMATest()
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
     * Test of supportsWeightedData method, of class ROMMA.
     */
    @Test
    public void testTrain_C()
    {
        System.out.println("supportsWeightedData");
        ROMMA nonAggro = new ROMMA();
        ROMMA aggro = new ROMMA();
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        nonAggro.setEpochs(1);
        nonAggro.trainC(train);
        
        aggro.setEpochs(1);
        aggro.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), aggro.classify(dpp.getDataPoint()).mostLikely());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), nonAggro.classify(dpp.getDataPoint()).mostLikely());
    }
}
