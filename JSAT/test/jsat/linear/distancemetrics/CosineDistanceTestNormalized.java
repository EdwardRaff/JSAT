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
import jsat.linear.DenseVector;
import jsat.linear.Vec;
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
public class CosineDistanceTestNormalized
{
    static private ExecutorService ex;
    static private Vec negOnes;
    static private Vec ones;
    static private Vec half;
    static private Vec inc;
    
    static private List<Vec> vecs;
    
    static private double[][] expected;
    
    public CosineDistanceTestNormalized()
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
        negOnes = new DenseVector(5);
        negOnes.mutableAdd(-1.0);
        
        ones = new DenseVector(5);
        ones.mutableAdd(1.0);
        
        half = new DenseVector(5);
        half.mutableAdd(0.5);
        
        inc = new DenseVector(5);
        for(int i = 0; i < inc.length(); i++)
            inc.set(i, i);
        
        vecs = Arrays.asList(negOnes, ones, half, inc);
        for(Vec v : vecs)
            v.normalize();
        //was computed as 1-x'y/(||x|| ||y||)
        expected = new double[][]
        {
            { 2.22044604925e-16 ,  2.0 ,  2.0 ,  1.81649658093 ,  },
            { 2.0 ,  2.22044604925e-16 ,  2.22044604925e-16 ,  0.183503419072 ,  },
            { 2.0 ,  2.22044604925e-16 ,  2.22044604925e-16 ,  0.183503419072 ,  },
            { 1.81649658093 ,  0.183503419072 ,  0.183503419072 ,  0.0 ,  },
        };
        //so lets get it to just x'y/(||x|| ||y||)
        
        for(int i = 0; i < expected.length; i++)
            for(int j = 0; j < expected[i].length; j++)
                expected[i][j] = (expected[i][j]*-1)+1;
        
    }
    
    @After
    public void tearDown()
    {
    }

    @Test
    public void testDist_Vec_Vec()
    {
        System.out.println("dist");
        
        CosineDistanceNormalized dist = new CosineDistanceNormalized();
        
        
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
        
        for (int i = 0; i < vecs.size(); i++)
            for (int j = 0; j < vecs.size(); j++)
            {
                CosineDistanceNormalized d = dist.clone();
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(vecs.get(i), vecs.get(j))), 1e-8);
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(i, j, vecs, cache)), 1e-8);
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(i, vecs.get(j), vecs, cache)), 1e-8);
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache)), 1e-8);
            }
        
        cache = null;
        for (int i = 0; i < vecs.size(); i++)
            for (int j = 0; j < vecs.size(); j++)
            {
                CosineDistanceNormalized d = dist.clone();
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(vecs.get(i), vecs.get(j))), 1e-8);
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(i, j, vecs, cache)), 1e-8);
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(i, vecs.get(j), vecs, cache)), 1e-8);
                assertEquals(expected[i][j], CosineDistance.distanceToCosine(d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache)), 1e-8);
            }
    }

    @Test
    public void testMetricProperties()
    {
        System.out.println("isSymmetric");
        CosineDistanceNormalized instance = new CosineDistanceNormalized();
        assertTrue(instance.isSymmetric());
        assertTrue(instance.isSubadditive());
        assertTrue(instance.isIndiscemible());
    }

    @Test
    public void testMetricBound()
    {
        System.out.println("metricBound");
        CosineDistanceNormalized instance = new CosineDistanceNormalized();
        assertTrue(instance.metricBound() > 0);
        assertEquals(1.0, instance.metricBound(), 0.0);
    }

    
}
