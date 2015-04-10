
package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class NHERDTest
{
    
    public NHERDTest()
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
    
    @Test
    public void testTrain_C()
    {
        System.out.println("train_C");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, new Random(132));
     
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random(231));

        for (NHERD.CovMode mode : NHERD.CovMode.values())
        {
            NHERD nherd0 = new NHERD(1, mode);
            nherd0.trainC(train);
            
            for (DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), nherd0.classify(dpp.getDataPoint()).mostLikely());
        }

    }

}
