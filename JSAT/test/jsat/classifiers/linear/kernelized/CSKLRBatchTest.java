package jsat.classifiers.linear.kernelized;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.svm.SupportVectorLearner;
import jsat.distributions.kernels.PukKernel;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
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
public class CSKLRBatchTest
{
    public CSKLRBatchTest()
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
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values())
        {
            CSKLRBatch instance = new CSKLRBatch(0.5, new RBFKernel(0.5), 10, mode, SupportVectorLearner.CacheMode.NONE);
            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
            ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.0);
        }
        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");

        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values())
        {
            CSKLRBatch instance = new CSKLRBatch(0.5, new RBFKernel(0.5), 10, mode, SupportVectorLearner.CacheMode.NONE);

            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
            ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.0);
        }

    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values())
        {
            CSKLRBatch instance = new CSKLRBatch(0.5, new RBFKernel(0.5), 10, mode, SupportVectorLearner.CacheMode.NONE);

            ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom());
            ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom(), 2.0, 10.0);

            instance = instance.clone();

            instance.trainC(t1);

            CSKLRBatch result = instance.clone();

            for (int i = 0; i < t1.getSampleSize(); i++)
                assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
            result.trainC(t2);

            for (int i = 0; i < t1.getSampleSize(); i++)
                assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

            for (int i = 0; i < t2.getSampleSize(); i++)
                assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
        }

    }

    @Test
    public void testSerializable_WithTrainedModel() throws Exception {
        System.out.println("Serializable");

        for(CSKLR.UpdateMode mode : CSKLR.UpdateMode.values()) {

            CSKLRBatch instance = new CSKLRBatch(0.5, new RBFKernel(0.5), 10, mode, SupportVectorLearner.CacheMode.NONE);

            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
            ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());

            instance.trainC(train);

            CSKLRBatch serializedBatch = serializeAndDeserialize(instance);

            for (int i = 0; i < test.getSampleSize(); i++)
                assertEquals(test.getDataPointCategory(i), serializedBatch.classify(test.getDataPoint(i)).mostLikely());
        }
    }

    private CSKLRBatch serializeAndDeserialize(CSKLRBatch batch) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(batch);

        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        ObjectInputStream ois = new ObjectInputStream(bais);
        return (CSKLRBatch) ois.readObject();
    }
}
