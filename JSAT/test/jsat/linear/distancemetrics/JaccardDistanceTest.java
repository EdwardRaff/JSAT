/*
 * Copyright (C) 2017 Edward Raff
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

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
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
public class JaccardDistanceTest
{
    
    public JaccardDistanceTest()
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
     * Test of dist method, of class JaccardDistance.
     */
    @Test
    public void testDist_Vec_Vec()
    {
        System.out.println("dist");
        Vec a = DenseVector.toDenseVec(0.0, 1.0, 0.0, 2.0, 5.0);
        Vec b = DenseVector.toDenseVec(1.0, 0.0, 0.0, 3.0, 0.0);
        double unweightedSim = 1.0/4;
        double weightedSim = 2.0/10;
        JaccardDistance j = new JaccardDistance(false);
        JaccardDistance wj = new JaccardDistance(true);
        
        assertEquals(unweightedSim, j.eval(a, b), 1e-12);
        assertEquals(1-unweightedSim, j.dist(a, b), 1e-12);
        assertEquals(unweightedSim, j.eval(b, a), 1e-12);
        assertEquals(1-unweightedSim, j.dist(b, a), 1e-12);
        
        assertEquals(weightedSim, wj.eval(a, b), 1e-12);
        assertEquals(1-weightedSim, wj.dist(a, b), 1e-12);
        assertEquals(weightedSim, wj.eval(b, a), 1e-12);
        assertEquals(1-weightedSim, wj.dist(b, a), 1e-12);
        
        assertEquals(1.0, j.eval(a, a), 1e-12);
        assertEquals(0.0, j.dist(a, a), 1e-12);
        assertEquals(1.0, j.eval(b, b), 1e-12);
        assertEquals(0.0, j.dist(b, b), 1e-12);
        
        assertEquals(1.0, wj.eval(a, a), 1e-12);
        assertEquals(0.0, wj.dist(a, a), 1e-12);
        assertEquals(1.0, wj.eval(b, b), 1e-12);
        assertEquals(0.0, wj.dist(b, b), 1e-12);
    }

    /**
     * Test of isSymmetric method, of class JaccardDistance.
     */
    @Test
    public void testIsSymmetric()
    {
        System.out.println("isSymmetric");
        JaccardDistance instance = new JaccardDistance();
        boolean expResult = true;
        boolean result = instance.isSymmetric();
        assertEquals(expResult, result);
    }

    /**
     * Test of isSubadditive method, of class JaccardDistance.
     */
    @Test
    public void testIsSubadditive()
    {
        System.out.println("isSubadditive");
        JaccardDistance instance = new JaccardDistance();
        boolean expResult = true;
        boolean result = instance.isSubadditive();
        assertEquals(expResult, result);
    }

    /**
     * Test of isIndiscemible method, of class JaccardDistance.
     */
    @Test
    public void testIsIndiscemible()
    {
        System.out.println("isIndiscemible");
        JaccardDistance instance = new JaccardDistance();
        boolean expResult = true;
        boolean result = instance.isIndiscemible();
        assertEquals(expResult, result);
    }

    /**
     * Test of metricBound method, of class JaccardDistance.
     */
    @Test
    public void testMetricBound()
    {
        System.out.println("metricBound");
        JaccardDistance instance = new JaccardDistance();
        double expResult = 1.0;
        double result = instance.metricBound();
        assertEquals(expResult, result, 0.0);
    }

    /**
     * Test of supportsAcceleration method, of class JaccardDistance.
     */
    @Test
    public void testSupportsAcceleration()
    {
        System.out.println("supportsAcceleration");
        JaccardDistance instance = new JaccardDistance();
        boolean expResult = false;
        boolean result = instance.supportsAcceleration();
        assertEquals(expResult, result);
    }

    /**
     * Test of getAccelerationCache method, of class JaccardDistance.
     */
    @Test
    public void testGetAccelerationCache_List()
    {
        System.out.println("getAccelerationCache");
        List<? extends Vec> vecs = null;
        JaccardDistance instance = new JaccardDistance();
        List<Double> expResult = null;
        List<Double> result = instance.getAccelerationCache(vecs);
        assertEquals(expResult, result);
    }

    /**
     * Test of getAccelerationCache method, of class JaccardDistance.
     */
    @Test
    public void testGetAccelerationCache_List_ExecutorService()
    {
        System.out.println("getAccelerationCache");
        List<? extends Vec> vecs = null;
        ExecutorService threadpool = null;
        JaccardDistance instance = new JaccardDistance();
        List<Double> expResult = null;
        List<Double> result = instance.getAccelerationCache(vecs, true);
        assertEquals(expResult, result);
    }


    /**
     * Test of getQueryInfo method, of class JaccardDistance.
     */
    @Test
    public void testGetQueryInfo()
    {
        System.out.println("getQueryInfo");
        Vec q = null;
        JaccardDistance instance = new JaccardDistance();
        List<Double> expResult = null;
        List<Double> result = instance.getQueryInfo(q);
        assertEquals(expResult, result);
    }

    /**
     * Test of normalized method, of class JaccardDistance.
     */
    @Test
    public void testNormalized()
    {
        System.out.println("normalized");
        JaccardDistance instance = new JaccardDistance();
        boolean expResult = true;
        boolean result = instance.normalized();
        assertEquals(expResult, result);
    }
    
}
