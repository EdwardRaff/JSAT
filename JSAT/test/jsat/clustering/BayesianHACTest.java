/*
 * This code was contributed under the Public Domain
 */
package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import jsat.DataSet;
import jsat.NormalClampedSample;
import jsat.SimpleDataSet;
import jsat.TestTools;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.ConstantVector;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.GridDataGenerator;
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
public class BayesianHACTest {
    
    public BayesianHACTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of log_exp_sum method, of class BayesianHAC.
     */
    @Test
    public void testLog_exp_sum() 
    {
        System.out.println("log_exp_sum");
        double log_a = 1.0;
        double log_b = 3.0;
        double expResult = Math.log(Math.exp(log_a)+Math.exp(log_b));
        double result = BayesianHAC.log_exp_sum(log_a, log_b);
        assertEquals(expResult, result, 1e-10);
        
    }

    @Test
    public void testBinaryClustering() 
    {
        System.out.println("cluster_BernoulliBeta");
        
        Random rand = RandomUtil.getRandom();
        
        int d = 5;
        SimpleDataSet sds = new SimpleDataSet(d, new CategoricalData[0]);
        
        //Hard coded test to correctly identify that there are two clusters
        
        for(int i = 0; i < 20; i++)
        {
            Vec x = DenseVector.random(d, rand).multiply(0.05);
            sds.add(new DataPoint(x));
        }
        
        for(int i = 0; i < 20; i++)
        {
            Vec x = DenseVector.random(d, rand).multiply(0.05).add(0.9);
            sds.add(new DataPoint(x));
        }
        
        BayesianHAC bhac = new BayesianHAC(BayesianHAC.Distributions.BERNOULLI_BETA);
        
        int[] designations = new int[sds.size()];
        bhac.cluster(sds, false, designations);
        
        //check both classes are homogonous
        for(int i = 1; i < 20; i++)
            assertEquals(designations[0], designations[i]);
        
        //check both classes are homogonous
        for(int i = 21; i < sds.size(); i++)
            assertEquals(designations[20], designations[i]);
        
        //Both classes have different values
        assertEquals(1, Math.abs(designations[0]-designations[20]));
        
//        for(int i = 0; i < designations.length; i++)
//            System.out.println(designations[i]);

    }
    
    @Test
    public void testClusterGuass() 
    {
        System.out.println("cluster_guass");
        
        Random rand = RandomUtil.getRandom();
        
        
        GridDataGenerator gdg = new GridDataGenerator(new NormalClampedSample(0, 0.05), rand, 2, 2);
        SimpleDataSet sds = gdg.generateData(10);

        for(BayesianHAC.Distributions cov_type : EnumSet.of(BayesianHAC.Distributions.GAUSSIAN_FULL, BayesianHAC.Distributions.GAUSSIAN_DIAG))
            for (boolean parallel : new boolean[]{ false})
            {
                BayesianHAC em = new BayesianHAC(cov_type);

                int[] designations = new int[sds.size()];
                em.cluster(sds, parallel);
                

                List<List<DataPoint>> grouped = ClustererBase.createClusterListFromAssignmentArray(designations, sds);
//                em = em.clone();

                TestTools.checkClusteringByCat(grouped);

//                for(int i = 0; i < designations.length; i++)
//                    System.out.println(designations[i]);
            }
    }
    
}
