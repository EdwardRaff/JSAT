
package jsat.datatransform;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for JL are inherently probabilistic, so occasional failures can be 
 * tolerated. 
 * 
 * @author Edward Raff
 */
public class JLTransformTest
{
    static DataSet ds;
    static double eps = 0.2;
    
    public JLTransformTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        List<DataPoint> dps = new ArrayList<DataPoint>(100);
        Random rand = RandomUtil.getRandom();
        
        for(int i = 0; i < 100; i++)
        {
            Vec v = DenseVector.random(2000, rand);
            dps.add(new DataPoint(v, new int[0], new CategoricalData[0]));
        }
        
        ds = new SimpleDataSet(dps);
        
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
     * Test of transform method, of class JLTransform.
     */
    @Test
    public void testTransform()
    {
        System.out.println("transform");
        Random rand = new XORWOW(124);
        int k = 550;
        
        List<Vec> transformed = new ArrayList<Vec>(ds.getSampleSize());
        for( JLTransform.TransformMode mode : JLTransform.TransformMode.values())
        {
            JLTransform jl = new JLTransform(k, mode, true);
            jl.fit(ds);

            transformed.clear();
            for(int i = 0; i < ds.getSampleSize(); i++)
                transformed.add(jl.transform(ds.getDataPoint(i)).getNumericalValues());
            
            int violations = 0;
            int count = 0;

            EuclideanDistance d = new EuclideanDistance();
            for(int i = 0; i < ds.getSampleSize(); i++)
            {
                DataPoint dpi = ds.getDataPoint(i);
                Vec vi = dpi.getNumericalValues();
                Vec vti = transformed.get(i);

                for(int j = i+1; j < ds.getSampleSize(); j++)
                {
                    count++;

                    DataPoint dpj = ds.getDataPoint(j);
                    Vec vj = dpj.getNumericalValues();
                    Vec vtj = transformed.get(j);

                    double trueDist = Math.pow(d.dist(vi, vj), 2);
                    double embDist = Math.pow(d.dist(vti, vtj), 2);

                    double err = (embDist-trueDist)/trueDist;
                    if( Math.abs(err) > eps)
                        violations++;
                }
            }

            assertTrue("Too many violations occured", violations < 150);
        }
        
        
    }
}
