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

package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.assertEquals;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class KDTreeTest
{
    static List<VectorCollection<Vec>> collectionFactories;
    
    public KDTreeTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        collectionFactories = new ArrayList<VectorCollection<Vec>>();
        for(KDTree.PivotSelection pivot : KDTree.PivotSelection.values())
            collectionFactories.add(new KDTree<>(pivot));
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
    public void testSearch_Vec_double()
    {
        System.out.println("search");
        Random rand = RandomUtil.getRandom();
        
        VectorArray<Vec> vecCol = new VectorArray<>(new EuclideanDistance());
        for(int i = 0; i < 2050; i++)
            vecCol.add(DenseVector.random(3, rand));
        
        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        for(VectorCollection<Vec> factory : collectionFactories)
        {
            VectorCollection<Vec> collection0 = factory.clone();
            collection0.build(vecCol, new EuclideanDistance());
            VectorCollection<Vec> collection1 = factory.clone();
            collection1.build(true, vecCol, new EuclideanDistance());
            
            collection0 = collection0.clone();
            collection1 = collection1.clone();
            
            for(int iters = 0; iters < 10; iters++)
                for(double range : new double[]{0.001, 0.1, 0.25, 0.5})
                {
                    int randIndex=  rand.nextInt(vecCol.size());

                    List<? extends VecPaired<Vec, Double>> foundTrue = vecCol.search(vecCol.get(randIndex), range);
                    List<? extends VecPaired<Vec, Double>> foundTest0 = collection0.search(vecCol.get(randIndex), range);
                    List<? extends VecPaired<Vec, Double>> foundTest1 = collection1.search(vecCol.get(randIndex), range);
                    
                    VectorArray<VecPaired<Vec, Double>>  testSearch0 = new VectorArray<>(new EuclideanDistance(), foundTest0);
                    assertEquals(factory.getClass().getName() + " failed", foundTrue.size(), foundTest0.size());
                    for(Vec v : foundTrue)
                    {
                        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch0.search(v, 1);
                        assertTrue(factory.getClass().getName() + " failed", nn.get(0).equals(v, 1e-13));
                    }
                    
                    
                    
                    VectorArray<VecPaired<Vec, Double>>  testSearch1 = new VectorArray<>(new EuclideanDistance(), foundTest1);
                    assertEquals(factory.getClass().getName() + " failed", foundTrue.size(), foundTest1.size());
                    for(Vec v : foundTrue)
                    {
                        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch1.search(v, 1);
                        assertTrue(factory.getClass().getName() + " failed", nn.get(0).equals(v, 1e-13));
                    }
                    

                }
        }
        
        ex.shutdownNow();
    }
    
    @Test
    public void testSearch_Vec_int()
    {
        System.out.println("search");
        Random rand = RandomUtil.getRandom();
        
        VectorArray<Vec> vecCol = new VectorArray<Vec>(new EuclideanDistance());
        for(int i = 0; i < 2050; i++)
            vecCol.add(DenseVector.random(3, rand));
        
        for(VectorCollection<Vec> factory : collectionFactories)
        {
            ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
            
            VectorCollection<Vec> collection0 = factory.clone();
            collection0.build(vecCol, new EuclideanDistance());
            VectorCollection<Vec> collection1 = factory.clone();
            collection1.build(true, vecCol, new EuclideanDistance());
            
            collection0 = collection0.clone();
            collection1 = collection1.clone();
            
            ex.shutdownNow();
            
            for(int iters = 0; iters < 10; iters++)
                for(int neighbours : new int[]{1, 2, 4, 10, 20})
                {
                    int randIndex=  rand.nextInt(vecCol.size());

                    List<? extends VecPaired<Vec, Double>> foundTrue = vecCol.search(vecCol.get(randIndex), neighbours);
                    List<? extends VecPaired<Vec, Double>> foundTest0 = collection0.search(vecCol.get(randIndex), neighbours);
                    List<? extends VecPaired<Vec, Double>> foundTest1 = collection1.search(vecCol.get(randIndex), neighbours);

                    VectorArray<VecPaired<Vec, Double>>  testSearch0 = new VectorArray<VecPaired<Vec, Double>>(new EuclideanDistance(), foundTest0);
                    assertEquals(factory.getClass().getName() + " failed", foundTrue.size(), foundTest0.size());
                    for(Vec v : foundTrue)
                    {
                        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch0.search(v, 1);
                        assertTrue(factory.getClass().getName() + " failed", nn.get(0).equals(v, 1e-13));
                    }
                    
                    VectorArray<VecPaired<Vec, Double>>  testSearch1 = new VectorArray<VecPaired<Vec, Double>>(new EuclideanDistance(), foundTest1);
                    assertEquals(factory.getClass().getName() + " failed " + neighbours, foundTrue.size(), foundTest1.size());
                    for(Vec v : foundTrue)
                    {
                        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch1.search(v, 1);
                        assertTrue(factory.getClass().getName() + " failed " + neighbours, nn.get(0).equals(v, 1e-13));
                    }
                    

                }
        }
        
    }
    
    @Test
    public void testSearch_Vec_int_incramental()
    {
        System.out.println("search");
        Random rand = RandomUtil.getRandom();
        
        VectorArray<Vec> vecCol = new VectorArray<>(new EuclideanDistance());
        for(int i = 0; i < 1000; i++)
            vecCol.add(DenseVector.random(3, rand));
        
        for(int leaf_size : new int[]{10, 40})
            for (KDTree.PivotSelection pm : KDTree.PivotSelection.values())
                {
                    KDTree<Vec> collection0 = new KDTree(pm);
                    collection0.setLeafSize(leaf_size);
                    for(Vec v : vecCol)
                        collection0.insert(v);

                    IntList trueNN = new IntList();
                    DoubleList trueNN_dists = new DoubleList();

                    IntList foundNN = new IntList();
                    DoubleList foundNN_dists = new DoubleList();
                    for(int iters = 0; iters < 10; iters++)
                        for(int neighbours : new int[]{1, 2, 5, 10, 20})
                        {
                            int randIndex=  rand.nextInt(vecCol.size());

                            Vec query = vecCol.get(randIndex);

                            vecCol.search(query, neighbours, trueNN, trueNN_dists);
                            collection0.search(query, neighbours, foundNN, foundNN_dists);

                            assertEquals(trueNN.size(), foundNN.size());
                            assertEquals(trueNN_dists.size(), foundNN_dists.size());

                            for (int i = 0; i < trueNN.size(); i++)
                            {
                                assertEquals(trueNN.get(i), foundNN.get(i));
                                assertEquals(trueNN_dists.get(i), trueNN_dists.get(i), 0.0);
                            }
                        }
                }
    }
}
