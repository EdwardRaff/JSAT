package jsat.classifiers.linear.kernelized;

import java.util.List;
import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.RBFKernel;
import jsat.parameters.Parameter;
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
public class CSKLRTest
{
    
    public CSKLRTest()
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
     * Test of classify method, of class CSKLR.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));
        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values())
        {
            CSKLR csklr = new CSKLR(0.5, new RBFKernel(0.5), 10, mode);
            csklr.trainC(trainSet);
            
            for(int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), csklr.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }

}
