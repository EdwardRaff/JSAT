/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.distributions.discrete;

import jsat.linear.Vec;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class BinomialTest
{
    static int[] testVals = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    public BinomialTest()
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
    public void testLogPmf()
    {
        System.out.println("logPmf");
        Binomial instance = new Binomial();
        
        double[] expected_7_5 = new double[]
        {
            -4.85203026391962,-2.90612011486430,-1.80750782619619,-1.29668220243020,
            -1.29668220243020,-1.80750782619619,-2.90612011486430,-4.85203026391962,
            -Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE
        };
        
        instance.setTrials(7);
        instance.setP(0.5);
        for(int i = 0; i < expected_7_5.length; i++)
        {
            assertEquals(expected_7_5[i], instance.logPmf(testVals[i]), 1e-4);
        }
        
    }

    @Test
    public void testPmf()
    {
        System.out.println("pmf");
        Binomial instance = new Binomial();
        
        double[] expected_7_5 = new double[]
        {
            0.00781250000000000, 0.0546875000000000, 0.164062500000000, 0.273437500000000, 0.273437500000000, 0.164062500000000, 0.0546875000000000, 0.00781250000000000, 0, 0, 0
        };
        
        instance.setTrials(7);
        instance.setP(0.5);
        for(int i = 0; i < expected_7_5.length; i++)
        {
            assertEquals(expected_7_5[i], instance.pmf(testVals[i]), 1e-4);
        }
    }

    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        Binomial instance = new Binomial();
        
        double[] expected_7_5 = new double[]
        {
            0.00781250000000000, 0.0625000000000000, 0.226562500000000, 0.500000000000000, 
            0.773437500000000, 0.937500000000000, 0.992187500000000, 1.00000000000000, 
            1.00000000000000, 1.00000000000000, 1.00000000000000
        };
        
        instance.setTrials(7);
        instance.setP(0.5);
        for(int i = 0; i < expected_7_5.length; i++)
        {
            assertEquals(expected_7_5[i], instance.cdf(testVals[i]), 1e-4);
            
            //its hard to get the right value for the probabilities right on the line, so lets nudge them a little to make sure we map to the right spot
            double val;
            if(i == 0)
                val = instance.invCdf(expected_7_5[i]*.99);
            else
                val = instance.invCdf(expected_7_5[i-1]+(expected_7_5[i]-expected_7_5[i-1])*0.95);
            
            double expected = testVals[i] >= instance.max() ? instance.max() : testVals[i];
            assertEquals(expected, val, 1e-3);
        }
    }

    @Test
    public void testSummaryStats()
    {
        System.out.println("stats");
        Binomial instance = new Binomial();
        //mean, median, variance, standard dev, skew
        double[] expected_7_5 = {3.50000000000000,3.00000000000000,1.75000000000000,1.32287565553230, 0};
        
        instance.setTrials(7);
        instance.setP(0.5);
        
        assertEquals(expected_7_5[0], instance.mean(), 1e-4);
        assertEquals(expected_7_5[1], instance.median(), 1e-4);
        assertEquals(expected_7_5[2], instance.variance(), 1e-4);
        assertEquals(expected_7_5[3], instance.standardDeviation(), 1e-4);
        assertEquals(expected_7_5[4], instance.skewness(), 1e-4);
        
    }

    
    @Test
    public void testSample()
    {
        System.out.println("sample");
        Binomial instance = new Binomial();
        
        instance.setTrials(7);
        instance.setP(0.5);
        
        //use odd number so median is always an actual int
        Vec samples = instance.sampleVec(10001, RandomUtil.getRandom());

        assertEquals(instance.mean(), samples.mean(), 2e-1);
    //  assertEquals(instance.median(), samples.median(), 1e-1);
        double median = samples.median();
        assertTrue(median == 3 || median == 4);//two vali values for this case
        assertEquals(instance.variance(), samples.variance(), 2e-1);
        assertEquals(instance.standardDeviation(), samples.standardDeviation(), 2e-1);
        assertEquals(instance.skewness(), samples.skewness(), 2e-1);
    }
}
