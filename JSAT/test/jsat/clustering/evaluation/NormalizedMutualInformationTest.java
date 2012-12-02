/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering.evaluation;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
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
public class NormalizedMutualInformationTest
{
    
    public NormalizedMutualInformationTest()
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
     * Test of evaluate method, of class NormalizedMutualInformation.
     */
    @Test
    public void testEvaluate_intArr_DataSet()
    {
        System.out.println("evaluate");
        ClassificationDataSet cds = new ClassificationDataSet(0, 
                new CategoricalData[]{new CategoricalData(2)}, 
                new CategoricalData(2));
        Vec emptyVec = new DenseVector(0);
        int[] clusterAssign = new int[8];
        clusterAssign[0] = 0; cds.addDataPoint(emptyVec, new int[]{0}, 0);
        clusterAssign[1] = 0; cds.addDataPoint(emptyVec, new int[]{0}, 0);
        clusterAssign[2] = 0; cds.addDataPoint(emptyVec, new int[]{0}, 0);
        clusterAssign[3] = 0; cds.addDataPoint(emptyVec, new int[]{1}, 1);
        clusterAssign[4] = 1; cds.addDataPoint(emptyVec, new int[]{1}, 1);
        clusterAssign[5] = 1; cds.addDataPoint(emptyVec, new int[]{1}, 1);
        clusterAssign[6] = 1; cds.addDataPoint(emptyVec, new int[]{1}, 1);
        clusterAssign[7] = 1; cds.addDataPoint(emptyVec, new int[]{1}, 1);
        
        //True NMI for this should be 0.14039740914097984
        NormalizedMutualInformation nmi = new NormalizedMutualInformation();
        assertEquals(1.0-0.14039740914097984, nmi.evaluate(clusterAssign, cds), 1e-13);
        
    }
}
