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
package jsat.distributions;

import jsat.linear.Vec;
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
public class TruncatedDistributionTest
{
    static TruncatedDistribution test;
    static double[] vals;
    
    public TruncatedDistributionTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        vals = new double[]
        {
            -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5
        };
    }
    
    @AfterClass
    public static void tearDownClass()
    {
    }
    
    @Before
    public void setUp()
    {
        test = new TruncatedDistribution(new Normal(-1, 2), 0, 3);
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of pdf method, of class TruncatedDistribution.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        
        double[] expectedVals = new double[]
        {
            0, 0, 0, 0.5268556714063524, 0.4233404250511103, 0.3195541179514373,
            0.2265978006626938, 0.1509466771109160, 0.09446001683895860, 0
        };
        
        for(int i = 0; i < vals.length; i++)
        {
            assertEquals(expectedVals[i], test.pdf(vals[i]), 1e-4);
        }
    }
    
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        
        double[] expectedVals = new double[]
        {
            0, 0, 0, 0.2866123013348931, 0.5244537766181539, 0.7099254909327465,
            0.8458397106527087, 0.9394339130936753, 1.000000000000000,
            1.000000000000000
        };
        
        for(int i = 0; i < vals.length; i++)
        {
            assertEquals(expectedVals[i], test.cdf(vals[i]), 1e-4);
        }
    }

    /**
     * Test of invCdf method, of class TruncatedDistribution.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        
        double[] expectedVals = new double[]
        {
            0.1659278188451017, 0.3403064608805872, 0.5255330999226814,
            0.7248810360893487, 0.9430469434588486, 1.187201184781877,
            1.469235993294530, 1.811386537666672, 2.264206423889610
        };

        for (int i = 0; i < expectedVals.length; i++)
        {
            assertEquals(expectedVals[i], test.invCdf((i+1)/10.0), 1e-4);
        }
    }
    
    @Test
    public void testMean()
    {
        System.out.println("mean");
        assertEquals(1.085986668284908, test.mean(), 1e-5);
    }

    @Test
    public void testVariance()
    {
        System.out.println("variance");
        assertEquals(0.6011260859550334, test.variance(), 1e-5);
    }

    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        assertEquals(0.5742049382125395, test.skewness(), 1e-5);
    }
    
    @Test
    public void testMedian()
    {
        System.out.println("median");
        assertEquals(0.9430469434588486, test.median(), 1e-5);
    }
    
    //Not sure I'm going to support this... well see
    @Test
    public void testMode()
    {
        System.out.println("mode");
        assertEquals(Math.nextUp(0.0), test.mode(), 1e-5);
    }

    /**
     * Test of min method, of class TruncatedDistribution.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        assertEquals(0, test.min(), 1e-16);//can't be zero, but should be very close to it!
    }

    /**
     * Test of max method, of class TruncatedDistribution.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        assertEquals(3, test.max(), 0.0);
    }
    
}
