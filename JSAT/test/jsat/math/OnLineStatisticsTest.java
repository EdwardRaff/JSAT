/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.math;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class OnLineStatisticsTest
{
    private final double[] data = new double[]
    {
        1.43725, 3.57142, 1.02601, 0.962941, 2.35466, 2.28253, 1.71812,
        2.92907, 0.707891, 0.0136063, 2.9936, 2.06371, 0.274257, 0.23791,
        0.0649932, 0.454671, 5.0087, 1.08846, 3.67667, 3.03826
    };
    private final double mean = 1.795236475;
    private final double variance = 1.885715208339471;//population variance
    private final double skewness = 0.5329150349287533;
    private final double kurt = -0.5903068502891378;
    private final double min = 0.0136063;
    private final double max = 5.0087;
    
    public OnLineStatisticsTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
    }

    /**
     * Test of add method, of class OnLineStatistics.
     */
    @Test
    public void testAdd_double()
    {
        System.out.println("add(double)");
        OnLineStatistics stats = new OnLineStatistics();
        for(double x :  data)
            stats.add(x);
        assertEquals(mean, stats.getMean(), 1e-8);
        assertEquals(variance, stats.getVarance(), 1e-8);
        assertEquals(skewness, stats.getSkewness(), 1e-8);
        assertEquals(kurt, stats.getKurtosis(), 1e-8);
        assertEquals(max, stats.getMax(), 1e-8);
        assertEquals(min, stats.getMin(), 1e-8);
    }
    
    @Test
    public void testAdd_double_double()
    {
        System.out.println("add(double, double)");
        OnLineStatistics stats = new OnLineStatistics();
        stats.add(10, 10);
        stats.add(100,1);
        assertEquals(200.0/11.0, stats.getMean(), 1e-10); 
        assertEquals(81000.0/121.0, stats.getVarance(), 1e-10); 
        assertEquals(9.0/Math.sqrt(10), stats.getSkewness(), 1e-10); 
        assertEquals(91.0/10.0-3.0, stats.getKurtosis(), 1e-10); 
        assertEquals(10, stats.getMin(), 0.0);
        assertEquals(100, stats.getMax(), 0.0);
    }
            

    /**
     * Test of add method, of class OnLineStatistics.
     */
    @Test
    public void testRemove_OnLineStatistics_OnLineStatistics()
    {
        System.out.println("remove");
        
        for(int j = 1; j < data.length-1; j++)
        {
            OnLineStatistics total = new OnLineStatistics();
            OnLineStatistics A = new OnLineStatistics();
            OnLineStatistics B = new OnLineStatistics();
        
            for(int i = 0; i < data.length; i++)
            {
                total.add(data[i]);
                if( i < j)
                    A.add(data[i]);
                else
                    B.add(data[i]);
            }
            OnLineStatistics stats = OnLineStatistics.remove(total, B);
            assertEquals(A.getSumOfWeights(), stats.getSumOfWeights(), 1e-8);
            assertEquals(A.getMean(), stats.getMean(), 1e-8);
            if(!Double.isNaN(A.getVarance()))
                assertEquals(A.getVarance(), stats.getVarance(), 1e-8);
            //skewness and kurtosis aren't stable any more
            //min & max are not possible
            
            
            stats = OnLineStatistics.remove(total, A);
            assertEquals(B.getSumOfWeights(), stats.getSumOfWeights(), 1e-8);
            assertEquals(B.getMean(), stats.getMean(), 1e-8);
            if(!Double.isNaN(B.getVarance()))
                assertEquals(B.getVarance(), stats.getVarance(), 1e-8);
        }
    }
    
    @Test
    public void testAdd_OnLineStatistics_OnLineStatistics()
    {
        System.out.println("add");
        
        for(int j = 1; j < data.length-1; j++)
        {
            
            OnLineStatistics A = new OnLineStatistics();
            OnLineStatistics B = new OnLineStatistics();
        
            for(int i = 0; i < data.length; i++)
            {
                if( i < j)
                    A.add(data[i]);
                else
                    B.add(data[i]);
            }
            OnLineStatistics stats = OnLineStatistics.add(A, B);
            assertEquals(mean, stats.getMean(), 1e-8);
            assertEquals(variance, stats.getVarance(), 1e-8);
            assertEquals(skewness, stats.getSkewness(), 1e-8);
            assertEquals(kurt, stats.getKurtosis(), 1e-1);//The kurtois is not numerically stable in this situation
            assertEquals(max, stats.getMax(), 1e-8);
            assertEquals(min, stats.getMin(), 1e-8);
        }
    }
}
