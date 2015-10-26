/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.datatransform.featureselection;

import jsat.classifiers.*;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static jsat.linear.DenseVector.*;
import jsat.linear.Vec;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class MutualInfoFSTest
{
    
    public MutualInfoFSTest()
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
    public void testSomeMethod()
    {
//        List<DataPoint> dps = new ArrayList<DataPoint>();
        
        final CategoricalData predicting = new CategoricalData(3);
        
        final CategoricalData[] catAtrs = new CategoricalData[]
        {
            new CategoricalData(3),
            new CategoricalData(3),
            new CategoricalData(2),//Info
            new CategoricalData(3)//Info
        };
        //Making numeric attributes at indecies 1 and 3 informative
        
        final ClassificationDataSet cds = new ClassificationDataSet(4, catAtrs, predicting);
        
        cds.addDataPoint(toDenseVec(0.0, 0.0, 1.0, 1.0), new int[]{0, 1, 0, 0}, 0);
        cds.addDataPoint(toDenseVec(1.0, 0.0, 0.0, 1.0), new int[]{1, 2, 0, 0}, 0);
        cds.addDataPoint(toDenseVec(0.0, 0.0, 1.0, 1.0), new int[]{2, 0, 0, 0}, 0);
        cds.addDataPoint(toDenseVec(1.0, 0.0, 0.0, 1.0), new int[]{0, 1, 0, 0}, 0);
        
        cds.addDataPoint(toDenseVec(1.0, 1.0, 0.0, 1.0), new int[]{1, 2, 0, 1}, 1);
        cds.addDataPoint(toDenseVec(0.0, 1.0, 1.0, 1.0), new int[]{2, 0, 0, 1}, 1);
        cds.addDataPoint(toDenseVec(1.0, 1.0, 0.0, 1.0), new int[]{0, 1, 0, 1}, 1);
        cds.addDataPoint(toDenseVec(0.0, 1.0, 1.0, 1.0), new int[]{1, 2, 0, 1}, 1);
        
        cds.addDataPoint(toDenseVec(0.0, 1.0, 1.0, 0.0), new int[]{2, 0, 1, 2}, 2);
        cds.addDataPoint(toDenseVec(1.0, 1.0, 0.0, 0.0), new int[]{0, 1, 1, 2}, 2);
        cds.addDataPoint(toDenseVec(0.0, 1.0, 1.0, 0.0), new int[]{1, 2, 1, 2}, 2);
        cds.addDataPoint(toDenseVec(1.0, 1.0, 0.0, 0.0), new int[]{2, 0, 1, 2}, 2);
        
        final MutualInfoFS minFS = new MutualInfoFS.MutualInfoFSFactory(4, MutualInfoFS.NumericalHandeling.BINARY).clone().getTransform(cds).clone();
        
        for(int i = 0; i < cds.getSampleSize(); i++)
        {
            final DataPoint dp =  cds.getDataPoint(i);
            
            final DataPoint trDp = minFS.transform(dp);
            
            final int[] origCat = dp.getCategoricalValues();
            final int[] tranCat = trDp.getCategoricalValues();
            
            final Vec origVals = dp.getNumericalValues();
            final Vec tranVals = trDp.getNumericalValues();
            
            assertEquals(origCat[2], tranCat[0]);
            assertEquals(origCat[3], tranCat[1]);
            
            assertEquals(origVals.get(1), tranVals.get(0), 0.0);
            assertEquals(origVals.get(3), tranVals.get(1), 0.0);
        }
        
        
    }
}
