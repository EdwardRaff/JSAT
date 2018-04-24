/*
 * Copyright (C) 2018 Edward Raff
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
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.random.RandomUtil;
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
public class DualTreeTest
{
    
    public DualTreeTest()
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
     * Test of search method, of class DualTree.
     */
    @Test
    public void testSearch_Radius_DT()
    {
        System.out.println("search_radius_dt");
        
        Random rand = RandomUtil.getRandom();
        
        
        VectorArray<Vec> A = new VectorArray<>(new EuclideanDistance());
        VectorArray<Vec> B = new VectorArray<>(new EuclideanDistance());
        for(int i = 0; i < 2500; i++)
        {
            A.add(DenseVector.random(3, rand));
            B.add(DenseVector.random(3, rand));
        }

        
        List<List<Integer>> nn_true = new ArrayList<>();
        List<List<Double>> dists_true = new ArrayList<>();
        
        double min_radius = 0.1;
        double max_radius = 0.2;
        
        for(int i = 0; i < B.size(); i++)
        {
            List<Integer> nn = new IntList();
            List<Double> dd = new DoubleList();
            
            A.search(B.get(i), max_radius, nn, dd);
            
            //remove everyone that was too close
            while(!dd.isEmpty() && dd.get(0) < min_radius)
            {
                nn.remove(0);
                dd.remove(0);           
            }
            
            nn_true.add(nn);
            dists_true.add(dd);
        }
        
        
//        VPTree<Vec> A_dt = new VPTree<>();
//        VPTree<Vec> B_dt = new VPTree<>();

//        for(DualTree<Vec> base : Arrays.asList(new SVPTree<>()))
//        for(DualTree<Vec> base : Arrays.asList(new VPTree<>()))
        for(boolean parallel : new boolean[]{false,true})
            for(DualTree<Vec> base : Arrays.asList(new BallTree<>(), new VPTree<>()))
            {
    //            System.out.println(base.getClass().getCanonicalName());
                DualTree<Vec> A_dt = base.clone();
                DualTree<Vec> B_dt = base.clone();
                A_dt.build(A);

                B_dt.build(B);

                List<List<Integer>> nn_found = new ArrayList<>();
                List<List<Double>> dists_found = new ArrayList<>();

                A_dt.search(B_dt, min_radius, max_radius, nn_found, dists_found, parallel);

                for(int i = 0; i < B.size(); i++)
                {
                    List<Integer> nn_t = nn_true.get(i);
                    List<Integer> nn_f = nn_found.get(i);

                    List<Double> dd_t = dists_true.get(i);
                    List<Double> dd_f = dists_found.get(i);


                    assertEquals(nn_t.size(), nn_f.size());
                    assertEquals(dd_t.size(), dd_f.size());

                    for(int j = 0; j < nn_t.size(); j++)
                    {
                        assertEquals(nn_t.get(j), nn_f.get(j));
                        assertEquals(dd_t.get(j), dd_f.get(j), 1e-10);
                    }
                }
            }
    }

    
    @Test
    public void testSearch_knn_DT()
    {
        System.out.println("search_knn_dt");
        
        Random rand = RandomUtil.getRandom();
        
        
        VectorArray<Vec> A = new VectorArray<>(new EuclideanDistance());
        VectorArray<Vec> B = new VectorArray<>(new EuclideanDistance());
        for(int i = 0; i < 2500; i++)
        {
            A.add(DenseVector.random(3, rand));
            B.add(DenseVector.random(3, rand));
        }

        
        List<List<Integer>> nn_true = new ArrayList<>();
        List<List<Double>> dists_true = new ArrayList<>();
        
        int K = 9;
        
        for(int i = 0; i < B.size(); i++)
        {
            List<Integer> nn = new IntList();
            List<Double> dd = new DoubleList();
            
            A.search(B.get(i), K, nn, dd);
            
            nn_true.add(nn);
            dists_true.add(dd);
        }
        
        
//        for(DualTree<Vec> base : Arrays.asList(new VPTree<>()))
//        for(DualTree<Vec> base : Arrays.asList(new SVPTree<>()))
        for(boolean parallel : new boolean[]{false, true})
            for(DualTree<Vec> base : Arrays.asList(new BallTree<>(), new VPTreeMV<>()))
            {
    //            System.out.println(base.getClass().getCanonicalName());
                DualTree<Vec> A_dt = base.clone();
                DualTree<Vec> B_dt = base.clone();
                A_dt.build(A);
                B_dt.build(B);

                List<List<Integer>> nn_found = new ArrayList<>();
                List<List<Double>> dists_found = new ArrayList<>();

                A_dt.search(B_dt, K, nn_found, dists_found, parallel);

                for(int i = 0; i < B.size(); i++)
                {
                    List<Integer> nn_t = nn_true.get(i);
                    List<Integer> nn_f = nn_found.get(i);

                    List<Double> dd_t = dists_true.get(i);
                    List<Double> dd_f = dists_found.get(i);

                    assertEquals(nn_t.size(), nn_f.size());
                    assertEquals(dd_t.size(), dd_f.size());

                    for(int j = 0; j < nn_t.size(); j++)
                    {
                        assertEquals(nn_t.get(j), nn_f.get(j));
                        assertEquals(dd_t.get(j), dd_f.get(j), 1e-10);
                    }
                }
            }
        
    }
    
}
