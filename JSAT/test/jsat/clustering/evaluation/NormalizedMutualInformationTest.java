/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
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
                new CategoricalData[]{}, 
                new CategoricalData(3));
        //Using example case from Manning's book http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
        Vec emptyVec = new DenseVector(0);
        int[] clusterAssign = new int[17];
        int X = 0, O = 1, D = 2;
        clusterAssign[0] = 0; cds.addDataPoint(emptyVec, X);
        clusterAssign[1] = 0; cds.addDataPoint(emptyVec, X);
        clusterAssign[2] = 0; cds.addDataPoint(emptyVec, X);
        clusterAssign[3] = 0; cds.addDataPoint(emptyVec, X);
        clusterAssign[4] = 0; cds.addDataPoint(emptyVec, X);
        clusterAssign[5] = 0; cds.addDataPoint(emptyVec, O);

        clusterAssign[6] = 1; cds.addDataPoint(emptyVec, X);
        clusterAssign[7] = 1; cds.addDataPoint(emptyVec, D);
        clusterAssign[8] = 1; cds.addDataPoint(emptyVec, O);
        clusterAssign[9] = 1; cds.addDataPoint(emptyVec, O);
        clusterAssign[10] = 1; cds.addDataPoint(emptyVec, O);
        clusterAssign[11] = 1; cds.addDataPoint(emptyVec, O);

        clusterAssign[12] = 2; cds.addDataPoint(emptyVec, X);
        clusterAssign[13] = 2; cds.addDataPoint(emptyVec, X);
        clusterAssign[14] = 2; cds.addDataPoint(emptyVec, D);
        clusterAssign[15] = 2; cds.addDataPoint(emptyVec, D);
        clusterAssign[16] = 2; cds.addDataPoint(emptyVec, D);
        
        //True NMI for this should be 0.36
        NormalizedMutualInformation nmi = new NormalizedMutualInformation();
        assertEquals(0.36, 1.0-nmi.evaluate(clusterAssign, cds), 1e-2);
        
    }
}
