
package jsat.datatransform;

import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import static java.lang.Math.abs;
import jsat.linear.*;
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
public class FastICATest
{
    public FastICATest()
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
     * Test the transform method with data of the same dimension
     */
    @Test
    public void testTransform2_2()
    {
        System.out.println("transform");
        
        SimpleDataSet source = new SimpleDataSet(new CategoricalData[0], 2);
        SimpleDataSet X = new SimpleDataSet(new CategoricalData[0], 2);
        
        Matrix mixing_true = new DenseMatrix(new double[][]
        {
            {2, -1.5}, 
            {0.5, 0}
        });
        
        DenseVector time = new DenseVector(200);
        for(int i = 0; i < time.length(); i++)
        {
            double t = i/(time.length()+0.0);
            time.set(i, t);
            Vec s = DenseVector.toDenseVec(Math.cos(4*t*3.14) , Math.sin(12*t*3.14));
            source.add(new DataPoint(s));
            X.add(new DataPoint(s.multiply(mixing_true.transpose())));
        }
        
        
        SimpleDataSet origX = X.shallowClone();
        
        FastICA ica = new FastICA(X, 2);
        
        X.applyTransform(ica);
        
        //make sure scales match up. Keep 0 as the axis around which the sign changes so comparisons work when ICA acidently gest the wrong sign
        LinearTransform linearX = new LinearTransform(X, -1, 1);
        LinearTransform linearS = new LinearTransform(source, -1, 1);
        
        X.applyTransform(linearX);
        source.applyTransform(linearS);
        
        //Lets go through and comapre our found components to truth. Check differnces in absolute value, becasue the independent compontents may have the wrong sign! 
        
        
        for(int found_c= 0; found_c < X.getNumNumericalVars(); found_c++)
        {
            Vec x_c = X.getNumericColumn(found_c);
            boolean found_match = false;
            //It has to match up to ONE of the true components
            SearchLoop:
            for(int true_c = 0; true_c < source.getNumNumericalVars(); true_c++)
            {
                Vec t_c = source.getNumericColumn(true_c);
                
                for(int i = 0; i < x_c.length(); i++)
                {
                    double cmp = abs(x_c.get(i))-abs(t_c.get(i));
                    if(abs(cmp) > 1e-3)
                        continue SearchLoop;
                }
                //we made it! 
                found_match = true;
            }
            
            if(!found_match)
                fail("The " + found_c + " component didn't match any of the true components");
        }
        
        X.applyTransform(new InverseOfTransform(linearX));
        source.applyTransform(new InverseOfTransform(linearS));
        
        
        X.applyTransform(new InverseOfTransform(ica));
        //make sure inverse maps back up to original data
        
        for(int inverted_c= 0; inverted_c < X.getNumNumericalVars(); inverted_c++)
        {
            Vec x_c = X.getNumericColumn(inverted_c);
            boolean found_match = false;
            //It has to match up to ONE of the true components
            SearchLoop:
            for(int true_x = 0; true_x < origX.getNumNumericalVars(); true_x++)
            {
                Vec t_c = origX.getNumericColumn(true_x);

                for(int i = 0; i < x_c.length(); i++)
                {
                    double cmp = abs(x_c.get(i))-abs(t_c.get(i));
                    if(abs(cmp) > 1e-3)
                        continue SearchLoop;
                }
                //we made it! 
                found_match = true;
            }
            
            if(!found_match)
                fail("The " + inverted_c + " component didn't match any of the true components");
        }
    }
    
    /**
     * Tests the transform method with data pre-whitened 
     */
    @Test
    public void testTransform2_2_prewhite()
    {
        System.out.println("transform");
        
        SimpleDataSet source = new SimpleDataSet(new CategoricalData[0], 2);
        SimpleDataSet X = new SimpleDataSet(new CategoricalData[0], 2);
        
        Matrix mixing_true = new DenseMatrix(new double[][]
        {
            {2, -1.5}, 
            {0.5, 1}
        });
        
        DenseVector time = new DenseVector(200);
        for(int i = 0; i < time.length(); i++)
        {
            double t = i/(time.length()+0.0);
            time.set(i, t);
            Vec s = DenseVector.toDenseVec(Math.cos(4*t*3.14) , Math.sin(12*t*3.14));
            source.add(new DataPoint(s));
            X.add(new DataPoint(s.multiply(mixing_true.transpose())));
        }
        ZeroMeanTransform zeroMean = new ZeroMeanTransform(X);
        X.applyTransform(zeroMean);
        WhitenedPCA whiten = new WhitenedPCA(X);
        X.applyTransform(whiten);
        
        SimpleDataSet origX = X.shallowClone();
        
        FastICA ica = new FastICA(X, 2, FastICA.DefaultNegEntropyFunc.LOG_COSH, true);
        
        X.applyTransform(ica);
        
        //make sure scales match up. Keep 0 as the axis around which the sign changes so comparisons work when ICA acidently gest the wrong sign
        LinearTransform linearX = new LinearTransform(X, -1, 1);
        LinearTransform linearS = new LinearTransform(source, -1, 1);
        
        X.applyTransform(linearX);
        source.applyTransform(linearS);
        
        //Lets go through and comapre our found components to truth. Check differnces in absolute value, becasue the independent compontents may have the wrong sign! 
        
        
        for(int found_c= 0; found_c < X.getNumNumericalVars(); found_c++)
        {
            Vec x_c = X.getNumericColumn(found_c);
            boolean found_match = false;
            //It has to match up to ONE of the true components
            SearchLoop:
            for(int true_c = 0; true_c < source.getNumNumericalVars(); true_c++)
            {
                Vec t_c = source.getNumericColumn(true_c);
                
                for(int i = 0; i < x_c.length(); i++)
                {
                    double cmp = abs(x_c.get(i))-abs(t_c.get(i));
                    if(abs(cmp) > 1e-3)
                        continue SearchLoop;
                }
                //we made it! 
                found_match = true;
            }
            
            if(!found_match)
                fail("The " + found_c + " component didn't match any of the true components");
        }
        
        X.applyTransform(new InverseOfTransform(linearX));
        source.applyTransform(new InverseOfTransform(linearS));
        
        
        
        X.applyTransform(new InverseOfTransform(ica));
        //make sure inverse maps back up to original data
        
        for(int inverted_c= 0; inverted_c < X.getNumNumericalVars(); inverted_c++)
        {
            Vec x_c = X.getNumericColumn(inverted_c);
            boolean found_match = false;
            //It has to match up to ONE of the true components
            SearchLoop:
            for(int true_x = 0; true_x < origX.getNumNumericalVars(); true_x++)
            {
                Vec t_c = origX.getNumericColumn(true_x);

                for(int i = 0; i < x_c.length(); i++)
                {
                    double cmp = abs(x_c.get(i))-abs(t_c.get(i));
                    if(abs(cmp) > 1e-3)
                        continue SearchLoop;
                }
                //we made it! 
                found_match = true;
            }
            
            if(!found_match)
                fail("The " + inverted_c + " component didn't match any of the true components");
        }
    }
    
    /**
     * Tests the transform method with data of a higher dimension
     */
    @Test
    public void testTransform2_3()
    {
        System.out.println("transform");

        SimpleDataSet source = new SimpleDataSet(new CategoricalData[0], 2);
        SimpleDataSet X = new SimpleDataSet(new CategoricalData[0], 3);
        
        Matrix mixing_true = new DenseMatrix(new double[][]
        {
            {2, 1.5, -1}, 
            {-0.5, 1, 2},
        });
        
        DenseVector time = new DenseVector(200);
        for(int i = 0; i < time.length(); i++)
        {
            double t = i/(time.length()+0.0);
            time.set(i, t);
            Vec s = DenseVector.toDenseVec(Math.cos(4*t*3.14) , Math.sin(12*t*3.14));
            source.add(new DataPoint(s));
            X.add(new DataPoint(mixing_true.transpose().multiply(s)));
        }

        SimpleDataSet origX = X.shallowClone();
        
        FastICA ica = new FastICA(X, 2);
        
        X.applyTransform(ica);
        
        //make sure scales match up. Keep 0 as the axis around which the sign changes so comparisons work when ICA acidently gest the wrong sign
        LinearTransform linearX = new LinearTransform(X, -1, 1);
        LinearTransform linearS = new LinearTransform(source, -1, 1);
        
        X.applyTransform(linearX);
        source.applyTransform(linearS);
        
        //Lets go through and comapre our found components to truth. Check differnces in absolute value, becasue the independent compontents may have the wrong sign! 
        
        
        for(int found_c= 0; found_c < X.getNumNumericalVars(); found_c++)
        {
            Vec x_c = X.getNumericColumn(found_c);
            boolean found_match = false;
            //It has to match up to ONE of the true components
            SearchLoop:
            for(int true_c = 0; true_c < source.getNumNumericalVars(); true_c++)
            {
                Vec t_c = source.getNumericColumn(true_c);
                
                for(int i = 0; i < x_c.length(); i++)
                {
                    double cmp = abs(x_c.get(i))-abs(t_c.get(i));
                    if(abs(cmp) > 1e-3)
                        continue SearchLoop;
                }
                //we made it! 
                found_match = true;
            }
            
            if(!found_match)
                fail("The " + found_c + " component didn't match any of the true components");
        }
        
        X.applyTransform(new InverseOfTransform(linearX));
        source.applyTransform(new InverseOfTransform(linearS));
        
        
        
        X.applyTransform(new InverseOfTransform(ica));
        //make sure inverse maps back up to original data
        
        for(int inverted_c= 0; inverted_c < X.getNumNumericalVars(); inverted_c++)
        {
            Vec x_c = X.getNumericColumn(inverted_c);
            boolean found_match = false;
            //It has to match up to ONE of the true components
            SearchLoop:
            for(int true_x = 0; true_x < origX.getNumNumericalVars(); true_x++)
            {
                Vec t_c = origX.getNumericColumn(true_x);

                for(int i = 0; i < x_c.length(); i++)
                {
                    double cmp = abs(x_c.get(i))-abs(t_c.get(i));
                    if(abs(cmp) > 1e-3)
                        continue SearchLoop;
                }
                //we made it! 
                found_match = true;
            }
            
            if(!found_match)
                fail("The " + inverted_c + " component didn't match any of the true components");
        }
    }
}
