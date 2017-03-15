/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.math;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.distributions.*;
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
public class ExponentialMovingStatisticsTest
{
    private List<ContinuousDistribution> distributions;
    public ExponentialMovingStatisticsTest()
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
        //EMA has difficulty with values centered about 0 and symetric distributoins
        distributions = new ArrayList<ContinuousDistribution>();
        distributions.add(new Normal(5, 1));
        distributions.add(new Normal(3, 7));
        distributions.add(new Normal(40, 7));
        distributions.add(new Exponential(0.1));
        distributions.add(new LogUniform(1, 100));
        distributions.add(new Uniform(-10, 10));
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of setSmoothing method, of class ExponentialMovingStatistics.
     */
    @Test
    public void testSetSmoothing()
    {
        System.out.println("setSmoothing");
        Random rand = RandomUtil.getRandom();

        for (ContinuousDistribution dist : distributions)
        {
            double smoothing = 0.01;
            ExponentialMovingStatistics instance = new ExponentialMovingStatistics(smoothing);
            Vec values = dist.sampleVec(100000, rand);

            for (double x : values.arrayCopy())
                instance.add(x);
            
//            System.out.println(dist.toString());
//            System.out.println(dist.mean() + " " + instance.getMean() + "   " + smoothing);
//            System.out.println(dist.standardDeviation() + " " + instance.getStandardDeviation() + "   " + smoothing);

            assertEquals(dist.mean(), instance.getMean(), Math.max(dist.mean() * 0.75, 3.5));
            assertEquals(dist.standardDeviation(), instance.getStandardDeviation(), Math.max(dist.mean() * 0.75, 1.5));
        }
    }
    
    @Test
    public void testSetSmoothingDrifted()
    {
        System.out.println("setSmoothing Drift");
        Random rand = RandomUtil.getRandom();

        for (ContinuousDistribution other_dist : distributions)
            for (ContinuousDistribution dist : distributions)
            {
                if(other_dist == dist)
                    continue;
                double smoothing = 0.01;
                ExponentialMovingStatistics instance = new ExponentialMovingStatistics(smoothing);
                

                //these first onces should eventually be forgoten
                for (double x : other_dist.sample(100000, rand))
                    instance.add(x);
                for (double x : dist.sample(100000, rand))
                    instance.add(x);

    //            System.out.println(dist.toString());
    //            System.out.println(dist.mean() + " " + instance.getMean() + "   " + smoothing);
    //            System.out.println(dist.standardDeviation() + " " + instance.getStandardDeviation() + "   " + smoothing);

                assertEquals(dist.mean(), instance.getMean(), Math.max(dist.mean() * 0.75, 3.5));
                assertEquals(dist.standardDeviation(), instance.getStandardDeviation(), Math.max(dist.mean() * 0.75, 1.5));
            }
    }


}
