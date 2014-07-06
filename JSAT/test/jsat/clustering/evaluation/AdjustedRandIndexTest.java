/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package jsat.clustering.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.Vec;
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
public class AdjustedRandIndexTest
{
    
    public AdjustedRandIndexTest()
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
     * Test of evaluate method, of class AdjustedRandIndex.
     */
    @Test
    public void testEvaluate_intArr_DataSet()
    {
        System.out.println("evaluate");
        //using example from http://www.otlet-institute.org/wikics/Clustering_Problems.html
        ClassificationDataSet cds = new ClassificationDataSet(1, new CategoricalData[0], new CategoricalData(3));
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                cds.addDataPoint(Vec.random(1), new int[0], i);
        int[] d = new int[9];
        d[0] = d[1] = 0;
        d[2] = d[3] = d[4] = d[5] = 1;
        d[6] = d[7] = 2;
        d[8] = 3;
        
        AdjustedRandIndex ari = new AdjustedRandIndex();
        double score = ari.evaluate(d, cds);
        //conver tot ARI
        score = 1.0-score;
        assertEquals(0.46, score, 0.005);
    }
    
}
