/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.trees;

import java.util.Arrays;

import jsat.distributions.Normal;

import java.util.Random;
import java.util.concurrent.Executors;

import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

import java.util.concurrent.ExecutorService;

import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPointPair;
import jsat.datatransform.NumericalToHistogram;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.utils.PairedReturn;

import org.junit.*;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class DecisionStumpTest
{
    static private ClassificationDataSet easyNumAtTrain;
    static private ClassificationDataSet easyNumAtTest;
    static private ClassificationDataSet easyCatAtTrain;
    static private ClassificationDataSet easyCatAtTest;
    static private ExecutorService ex;
    static private DecisionStump stump;
    
    public DecisionStumpTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
        stump = new DecisionStump();
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 1);
        easyNumAtTrain = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        easyNumAtTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        
        easyCatAtTrain = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        easyCatAtTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        NumericalToHistogram nth = new NumericalToHistogram(easyCatAtTrain, 2);
        easyCatAtTrain.applyTransform(nth);
        easyCatAtTest.applyTransform(nth);
        
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }

    @After
    public void tearDown() throws Exception
    {
    }

    /**
     * Test of threshholdSplit method, of class DecisionStump.
     */
    @Test
    public void testThreshholdSplit()
    {
        System.out.println("threshholdSplit");
        Distribution dist1 = new Normal(0, 1);
        Distribution dist2 = new Normal(3, 2);
        PairedReturn<Integer, Double> ret = DecisionStump.threshholdSplit(dist1, dist2);
        assertEquals(0, (int) ret.getFirstItem());
        assertEquals(1.418344988105127, ret.getSecondItem(), 1e-6);
        
        ret = DecisionStump.threshholdSplit(dist2, dist1);
        assertEquals(1, (int) ret.getFirstItem());
        assertEquals(1.418344988105127, ret.getSecondItem(), 1e-6);
        
        dist2 = new Normal(3, 1);
        ret = DecisionStump.threshholdSplit(dist2, dist1);
        assertEquals(1, (int) ret.getFirstItem());
        assertEquals(1.5, ret.getSecondItem(), 1e-6);
        
        try
        {
            dist1 = new Normal(0, 2);
            dist2 = new Normal(3, 10);
            ret = DecisionStump.threshholdSplit(dist2, dist1);
            assertEquals(1, (int) ret.getFirstItem());
            assertEquals(4.896411863121647, ret.getSecondItem(), 1e-6);//This is the spliting point, but we dont expect it to get it 
        }
        catch(ArithmeticException ex)
        {
            
        }
        
        dist1 = new Normal(0, 1);
        dist2 = new Normal(100, 1);
        ret = DecisionStump.threshholdSplit(dist1, dist2);
        assertEquals(0, (int) ret.getFirstItem());
        assertEquals(50, ret.getSecondItem(), 1e-6);
        
        dist1 = new Normal(0, 1);
        dist2 = new Normal(100, 1);
        ret = DecisionStump.threshholdSplit(dist2, dist1);
        assertEquals(1, (int) ret.getFirstItem());
        assertEquals(50, ret.getSecondItem(), 1e-6);
    }

    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC(ClassificationDataSet, ExecutorService)");
        stump.trainC(easyNumAtTrain, ex);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }

    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC(ClassificationDataSet)");
        stump.trainC(easyNumAtTrain);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }

    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_List_Set()
    {
        System.out.println("trainC(List<DataPointPair>, Set<integer>)");
        stump.setPredicting(easyNumAtTrain.getPredicting());
        stump.trainC(easyNumAtTrain.getAsDPPList(), new IntSet(Arrays.asList(0)));
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }

    /**
     * Test of supportsWeightedData method, of class DecisionStump.
     */
    @Test
    public void testSupportsWeightedData()
    {
        System.out.println("supportsWeightedData");
        assertTrue(stump.supportsWeightedData());
    }

    /**
     * Test of clone method, of class DecisionStump.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        Classifier clone = stump.clone();
        clone.trainC(easyNumAtTrain);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), clone.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
        try
        {
            stump.classify(easyNumAtTest.getDataPoint(0));
            fail("Stump should not have been trained");
        }
        catch(Exception ex )
        {
            
        }
        clone = null;
        stump.trainC(easyNumAtTrain);
        clone = stump.clone();
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), clone.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }
    
    @Test
    public void testIntersection(){
    	Distribution d = new Uniform(1, 2);
    	//XXX create the test for double comparison
    	DecisionStump.intersections(Arrays.asList(new Distribution[]{d}));
    }

   
    @Test
    public void testInfoGainSplit()
    {
        System.out.println("testInfoGainSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testInfoGainRatioSplit()
    {
        System.out.println("testInfoGainRatioSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN_RATIO);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN_RATIO);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testGiniSplit()
    {
        System.out.println("testGiniSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.GINI);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.GINI);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testNumericCKDEInter()
    {
        System.out.println("testNumericCKDEInter");
        
        DecisionStump instance = new DecisionStump();
        instance.setNumericHandling(DecisionStump.NumericHandlingC.PDF_INTERSECTIONS);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }

    @Test
    public void testNumericCBinary()
    {
        System.out.println("testNumericCBinary");
        
        DecisionStump instance = new DecisionStump();
        instance.setNumericHandling(DecisionStump.NumericHandlingC.BINARY_BEST_GAIN);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
   
}
