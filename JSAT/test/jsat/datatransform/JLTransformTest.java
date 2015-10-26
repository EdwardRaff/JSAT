
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
    static double eps = 0.15;
    
    public JLTransformTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        final List<DataPoint> dps = new ArrayList<DataPoint>(100);
        final Random rand = new Random();
        
        for(int i = 0; i < 100; i++)
        {
            final Vec v = DenseVector.random(2000, rand);
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
        final Random rand = new Random(124);
        final int k = 550;
        
        final List<Vec> transformed = new ArrayList<Vec>(ds.getSampleSize());
        for( final JLTransform.TransformMode mode : JLTransform.TransformMode.values())
        {
            final JLTransform jl = new JLTransform(k, ds.getNumNumericalVars(), mode, rand, true);

            transformed.clear();
            for(int i = 0; i < ds.getSampleSize(); i++) {
              transformed.add(jl.transform(ds.getDataPoint(i)).getNumericalValues());
            }
            
            int violations = 0;
            int count = 0;

            final EuclideanDistance d = new EuclideanDistance();
            for(int i = 0; i < ds.getSampleSize(); i++)
            {
                final DataPoint dpi = ds.getDataPoint(i);
                final Vec vi = dpi.getNumericalValues();
                final Vec vti = transformed.get(i);

                for(int j = i+1; j < ds.getSampleSize(); j++)
                {
                    count++;

                    final DataPoint dpj = ds.getDataPoint(j);
                    final Vec vj = dpj.getNumericalValues();
                    final Vec vtj = transformed.get(j);

                    final double trueDist = Math.pow(d.dist(vi, vj), 2);
                    final double embDist = Math.pow(d.dist(vti, vtj), 2);

                    final double err = (embDist-trueDist)/trueDist;
                    if( Math.abs(err) > eps) {
                      violations++;
                    }
                }
            }

            assertTrue("Too many violations occured", violations < 150);
        }
        
        
    }
}
