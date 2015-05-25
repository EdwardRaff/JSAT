/*
 * Copyright (C) 2015 Edward Raff
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
package jsat.linear.distancemetrics;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.linear.*;
import jsat.utils.SystemInfo;
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
public class MinkowskiDistanceTest
{
    static private ExecutorService ex;
    static private Vec zero;
    static private Vec ones;
    static private Vec half;
    static private Vec inc;
    
    static private List<Vec> vecs;
    
    static private double[][] expectedP2;
    static private double[][] expectedP1;
    static private double[][] expectedP1p5;
    
    public MinkowskiDistanceTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        ex.shutdown();
    }
    
    @Before
    public void setUp()
    {
        zero = new DenseVector(5);
        
        ones = new DenseVector(5);
        ones.mutableAdd(1.0);
        
        half = new DenseVector(5);
        half.mutableAdd(0.5);
        
        inc = new DenseVector(5);
        for(int i = 0; i < inc.length(); i++)
            inc.set(i, i);
        
        vecs = Arrays.asList(zero, ones, half, inc);
        expectedP2 = new double[][]
        {
            { 0.0 ,  2.2360679775 ,  1.11803398875 ,  5.47722557505 ,  },
            { 2.2360679775 ,  0.0 ,  1.11803398875 ,  3.87298334621 ,  },
            { 1.11803398875 ,  1.11803398875 ,  0.0 ,  4.60977222865 ,  },
            { 5.47722557505 ,  3.87298334621 ,  4.60977222865 ,  0.0 ,  },
        };
        
        expectedP1 = new double[][]
        {
            { 0 ,  5 ,  2.5 ,  10 ,  },
            { 5 ,  0 ,  2.5 ,  7 ,  },
            { 2.5 ,  2.5 ,  0.0 ,  8.5 ,  },
            { 10 ,  7 ,  8.5 ,  0 ,  },
        };
        
        expectedP1p5 = new double[][]
        {
            { 0.0 ,  2.92401773821 ,  1.46200886911 ,  6.61786032327 ,  },
            { 2.92401773821 ,  0.0 ,  1.46200886911 ,  4.64919159806 ,  },
            { 1.46200886911 ,  1.46200886911 ,  0.0 ,  5.54151812966 ,  },
            { 6.61786032327 ,  4.64919159806 ,  5.54151812966 ,  0.0 ,  },
        };
    }
    
    @After
    public void tearDown()
    {
    }

    @Test
    public void testDist_Vec_Vec()
    {
        System.out.println("dist");
        
        MinkowskiDistance dist = new MinkowskiDistance(2.5);
        
        List<Double> cache = dist.getAccelerationCache(vecs);
        List<Double> cache2 = dist.getAccelerationCache(vecs, ex);
        if(cache != null)
        {
            assertEquals(cache.size(), cache2.size());
            for(int i = 0; i < cache.size(); i++)
                assertEquals(cache.get(i), cache2.get(i), 0.0);
            assertTrue(dist.supportsAcceleration());
        }
        else
        {
            assertNull(cache2);
            assertFalse(dist.supportsAcceleration());
        }
        
        try
        {
            dist.dist(half, new DenseVector(half.length()+1));
            fail("Distance between vecs should have erred");
        }
        catch (Exception ex)
        {

        }
        
        for(int rounds = 0; rounds < 3; rounds++)
        {
            //some code so that dense on dense, dense on sparse, and sparse on sparse all get run
            if(rounds == 1)
                for(int i = 0; i <vecs.size(); i+=2)
                    vecs.set(i, new SparseVector(vecs.get(i)));
            else if(rounds == 2)
                for(int i = 1; i <vecs.size(); i+=2)
                    vecs.set(i, new SparseVector(vecs.get(i)));
            
            for (int i = 0; i < vecs.size(); i++)
                for (int j = 0; j < vecs.size(); j++)
                {
                    MinkowskiDistance d = dist.clone();

                    d.setP(2.0);
                    assertEquals(2.5, dist.getP(), 0.0);

                    assertEquals(expectedP2[i][j], d.dist(vecs.get(i), vecs.get(j)), 1e-8);
                    assertEquals(expectedP2[i][j], d.dist(i, j, vecs, cache), 1e-8);
                    assertEquals(expectedP2[i][j], d.dist(i, vecs.get(j), vecs, cache), 1e-8);
                    assertEquals(expectedP2[i][j], d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache), 1e-8);

                    d.setP(1.0);
                    assertEquals(expectedP1[i][j], d.dist(vecs.get(i), vecs.get(j)), 1e-8);
                    assertEquals(expectedP1[i][j], d.dist(i, j, vecs, cache), 1e-8);
                    assertEquals(expectedP1[i][j], d.dist(i, vecs.get(j), vecs, cache), 1e-8);
                    assertEquals(expectedP1[i][j], d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache), 1e-8);

                    d.setP(1.5);
                    assertEquals(expectedP1p5[i][j], d.dist(vecs.get(i), vecs.get(j)), 1e-8);
                    assertEquals(expectedP1p5[i][j], d.dist(i, j, vecs, cache), 1e-8);
                    assertEquals(expectedP1p5[i][j], d.dist(i, vecs.get(j), vecs, cache), 1e-8);
                    assertEquals(expectedP1p5[i][j], d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache), 1e-8);
                }
        }
    }

    @Test
    public void testMetricProperties()
    {
        System.out.println("isSymmetric");
        EuclideanDistance instance = new EuclideanDistance();
        assertTrue(instance.isSymmetric());
        assertTrue(instance.isSubadditive());
        assertTrue(instance.isIndiscemible());
    }

    @Test
    public void testMetricBound()
    {
        System.out.println("metricBound");
        EuclideanDistance instance = new EuclideanDistance();
        assertTrue(instance.metricBound() > 0);
        assertTrue(Double.isInfinite(instance.metricBound()));
    }

    
}
