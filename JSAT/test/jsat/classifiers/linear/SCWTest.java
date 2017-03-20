
package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
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
public class SCWTest
{
    
    public SCWTest()
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
    public void testTrainC_Full()
    {
        System.out.println("TrainC_Full");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
     
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

        for (SCW.Mode mode : SCW.Mode.values())
        {   
            SCW scwFull = new SCW(0.9, mode, false);
            scwFull.trainC(train);
            
            for (DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), scwFull.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test
    public void testTrainC_Diag()
    {
        System.out.println("TrainC_Diag");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
     
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

        for (SCW.Mode mode : SCW.Mode.values())
        {   
            SCW scwDiag = new SCW(0.9, mode, true);
            scwDiag.trainC(train);
            
            for (DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), scwDiag.classify(dpp.getDataPoint()).mostLikely());
        }
    }
}
