/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.utils.random.RandomUtil;
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
public class ZipfTest
{
    static  Zipf Zipf_20k_half = null;
    static  Zipf Zipf_20k_2 = null;
    static  Zipf Zipf_inf_half = null;
    static  Zipf Zipf_inf_2 = null;
    
    static Zipf[] Zs = null;
    
    static final int[] xs = new int[]{0,1,2,3,4,5,10,100,1000,10000,20000,20001};
    
    public ZipfTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        Zipf_20k_half = new Zipf(20000, 0.5);
        Zipf_20k_2 = new Zipf(20000, 2);
        Zipf_inf_half = new Zipf(0.5);
        Zipf_inf_2 = new Zipf(2);
        Zs = new Zipf[]
        {
            Zipf_20k_half,
            Zipf_20k_2,
            Zipf_inf_half,
            Zipf_inf_2,
        };
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
     * Test of setCardinality method, of class Zipf.
     */
    @Test
    public void testGetCardinality()
    {
        System.out.println("setCardinality");
        assertEquals(20000, Zipf_20k_half.getCardinality(), 0);
        assertEquals(20000, Zipf_20k_2.getCardinality(), 0);
        assertEquals(Double.POSITIVE_INFINITY, Zipf_inf_half.getCardinality(), 0);
        assertEquals(Double.POSITIVE_INFINITY, Zipf_inf_2.getCardinality(), 0);
    }


    /**
     * Test of pmf method, of class Zipf.
     */
    @Test
    public void testPmf()
    {
        System.out.println("pmf");
        double[][] expected = new double[][]
        {
            {0,0.38487689516722987426,0.13607453124738610253,0.074069593009889181126,0.048109611895903734283,0.034424436022519451529,0.012170876075022983100,0.00038487689516722987426,0.000012170876075022983100,3.8487689516722987426e-7,1.3607453124738610253e-7,0},
            {0,0.83190737344575156199,0.10398842168071894525,0.030811384201694502296,0.012998552710089868156,0.0066552589875660124959,0.00083190737344575156199,8.3190737344575156199e-7,8.3190737344575156199e-10,8.3190737344575156199e-13,1.0398842168071894525e-13,0},
            {0,0.38279338399942656225,0.13533789880967029021,0.073668621098692236550,0.047849172999928320281,0.034238081118395924455,0.012104989666816425649,0.00038279338399942656225,0.000012104989666816425649,3.8279338399942656225e-7,1.3533789880967029021e-7,1.3532774910161896123e-7},
            {0,0.83190737258070746868,0.10398842157258843359,0.030811384169655832173,0.012998552696573554198,0.0066552589806456597495,0.00083190737258070746868,8.3190737258070746868e-7,8.3190737258070746868e-10,8.3190737258070746868e-13,1.0398842157258843359e-13,1.0397282486904889313e-13}
        };
        
        for(int z = 0; z < Zs.length; z++)
        {
            Zipf zipf = Zs[z];
            for(int i = 0; i < xs.length; i++)
            {
                double ex = expected[z][i];
                double acc = zipf.pmf(xs[i]);
                double err;
                if(ex == 0)
                    err = ex-acc;
                else
                    err = (ex-acc)/ex;
                assertEquals(0, err, 1e-3);
            }
        }
    }

    /**
     * Test of cdf method, of class Zipf.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        double[][] expected = new double[][]
        {
            {0,0.38487689516722987426,0.52095142641461597679,0.59502101942450515792,0.64313063132040889220,0.67755506734292834373,0.76795891437272322068,0.92865949153851453895,0.98110724498011254401,0.99774556774377231118,1.0000000000000000000,1.0000000000000000000},
            {0,0.83190737344575156199,0.93589579512647050724,0.96670717932816500953,0.97970573203825487769,0.98636099102582089019,0.99623568881949349504,0.99995881954514754426,0.99999958550189121097,0.99999999688071128956,1.0000000000000000000,1.0000000000000000000},
            {0,0.38279338399942656225,0.51813128280909685246,0.59179990390778908901,0.63964907690771740929,0.67388715802611333375,0.76380160850531216055,0.92363224140736199479,0.97579607164807706923,0.99234432371191855118,0.99458655171571673136,0.99458668704346583298},
            {0,0.83190737258070746868,0.93589579415329590227,0.96670717832295173444,0.97970573101952528864,0.98636099000017094839,0.99623568778357552428,0.99995881850535814221,0.99999958446205941916,0.99999999584087906999,0.99999999896016777719,0.99999999896027175001}
        };
        
        for(int z = 0; z < Zs.length; z++)
        {
            Zipf zipf = Zs[z];
            for(int i = 0; i < xs.length; i++)
            {
                double ex = expected[z][i];
                double acc = zipf.cdf(xs[i]);
                double err;
                if(ex == 0)
                    err = ex-acc;
                else
                    err = (ex-acc)/ex;
                assertEquals(0, err, 1e-3);
                ///and test inverse CDF while we are here
//                System.out.println(xs[i] + ", " + zipf.invCdf(ex) + ", " + zipf.invCdf(acc));
                assertEquals(xs[i], zipf.invCdf(ex), 1.0);//I'll let you be off by one...
            }
        }
    }

    /**
     * Test of mean method, of class Zipf.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        double[] expected = new double[]
        {
            108.29892902835748493,1.3683911847143409720,Double.POSITIVE_INFINITY,1.3684327776202058757
        };
        
        for(int i = 0; i < Zs.length; i++)
            assertEquals(expected[i], Zs[i].mean(), 1e-3);
    }

    /**
     * Test of mode method, of class Zipf.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        double[] expected = new double[]
        {
            1, 1, 1, 1
        };
        
        for(int i = 0; i < Zs.length; i++)
            assertEquals(expected[i], Zs[i].mode(), 1e-3);
    }

    /**
     * Test of variance method, of class Zipf.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        double[] expected = new double[]
        {
            714029.31029974343795,6.8465006485901065799,Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY
        };
        
        for(int i = 0; i < Zs.length; i++)
            assertEquals(expected[i], Zs[i].variance(), 1e-3);
        
        assertEquals(0.13066023147512419133, new Zipf(3.5).variance(), 1e-3);
    }

    /**
     * Test of skewness method, of class Zipf.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        double[] expected = new double[]
        {
            14.048173897535615783,927.04482492331320690,Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY
        };
        
        for(int i = 0; i < Zs.length; i++)
            assertEquals(expected[i], Zs[i].skewness(), 1e-3);
        
        assertEquals(17.763027886751897884, new Zipf(3.5).skewness(), 1e-3);
    }

    /**
     * Test of min method, of class Zipf.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        double[] expected = new double[]
        {
            1, 1, 1, 1
        };
        
        for(int i = 0; i < Zs.length; i++)
            assertEquals(expected[i], Zs[i].min(), 0);
    }

    /**
     * Test of max method, of class Zipf.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        double[] expected = new double[]
        {
            20000, 20000, Double.POSITIVE_INFINITY,Double.POSITIVE_INFINITY
        };
        
        for(int i = 0; i < Zs.length; i++)
            assertEquals(expected[i], Zs[i].max(), 0);
    }
    
    @Test
    public void testSampling()
    {
        System.out.println("max");
        
        Random rand =RandomUtil.getRandom();
        //only test the 2 cases where mean and variance are well defined
        for(int i = 0; i < Zs.length/2; i++)
        {
            Zipf zipf = Zs[i];
            
            DenseVector samples = zipf.sampleVec(10000, rand);
            double relErrMean = (zipf.mean()-samples.mean())/zipf.mean();
            assertEquals(0, relErrMean, 0.1);
            double relErrStd = (zipf.standardDeviation()-samples.standardDeviation())/zipf.standardDeviation();
            assertEquals(0, relErrStd, 0.5);
        }
    }
    
}
