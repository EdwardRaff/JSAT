/*
 *  This code contributed under the Public Domain
 */
package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import jsat.NormalClampedSample;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
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
 * @author edwardraff
 */
public class VBGMMTest {
    
    public VBGMMTest() {
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
     * Test of cluster method, of class VBGMM.
     */
    @Test
    public void testCluster() 
    {
        System.out.println("cluster");
        
        
        GridDataGenerator gdg = new GridDataGenerator(new NormalClampedSample(0, 0.05), RandomUtil.getRandom(), 2, 2);
        SimpleDataSet easyData = gdg.generateData(500);

        for (boolean parallel : new boolean[]{true, false})
        {
            VBGMM em = new VBGMM();

            List<List<DataPoint>> clusters = em.cluster(easyData, parallel);
            assertEquals(4, clusters.size());
            
            em = em.clone();

            List<Vec> means = Arrays.stream(em.normals).map(n->n.getMean()).collect(Collectors.toList());
            //we should have 1 mean at each of the coordinates of our 2x2 grid
            //(0,0), (0,1), (1,0), (1,1)

            List<Vec> expectedMeans = new ArrayList<>();
            expectedMeans.add(DenseVector.toDenseVec(0,0));
            expectedMeans.add(DenseVector.toDenseVec(0,1));
            expectedMeans.add(DenseVector.toDenseVec(1,0));
            expectedMeans.add(DenseVector.toDenseVec(1,1));

            for(Vec expected : expectedMeans)
                assertEquals(1, means.stream().filter(f->f.subtract(expected).pNorm(2) < 0.05).count());
        }
    }
    
}
