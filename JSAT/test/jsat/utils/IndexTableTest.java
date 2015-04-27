/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import java.util.List;
import java.util.Arrays;
import java.util.Comparator;
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
public class IndexTableTest
{
    static final double[] array = new double[] {9.0, 4.0, 3.0, 2.0, 1.0, 10.0, 11.0 };
    static final Double[] arrayD = new Double[] {9.0, 4.0, 3.0, 2.0, 1.0, 10.0, 11.0 };
    static final List<Double> list = Arrays.asList(arrayD);
    
    public IndexTableTest()
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
    
    @After
    public void tearDown()
    {
    }

    
    @Test
    public void testSortD()
    {
        IndexTable idt = new IndexTable(array);
        for(int i = 0; i < idt.length()-1; i++)
            assertTrue(array[idt.index(i)] <= array[idt.index(i+1)]);
    }
    
    @Test
    public void testSortG()
    {
        IndexTable idt = new IndexTable(arrayD);
        for(int i = 0; i < idt.length()-1; i++)
            assertTrue(arrayD[idt.index(i)].compareTo(arrayD[idt.index(i+1)]) <= 0);
    }
    
    @Test
    public void testSortList()
    {
        IndexTable idt = new IndexTable(list);
        for(int i = 0; i < idt.length()-1; i++)
            assertTrue(list.get(idt.index(i)).compareTo(list.get(idt.index(i+1))) <= 0);
    }
    
    @Test
    public void testSortListComparator()
    {
        IndexTable idt = new IndexTable(list, new Comparator<Double>() {

            @Override
            public int compare(Double o1, Double o2)
            {
                return -o1.compareTo(o2);
            }
        });
        for(int i = 0; i < idt.length()-1; i++)
            assertTrue(list.get(idt.index(i)).compareTo(list.get(idt.index(i+1))) >= 0);
    }
    
    @Test
    public void testApply_double()
    {
        IndexTable idt = new IndexTable(array);
        double[] test = Arrays.copyOf(array, array.length);
        idt.apply(test);
        for(int i = 0; i < test.length-1; i++)
            assertTrue(test[i] <= test[i+1]);
    }
    
    @Test
    public void testApply_List()
    {
        IndexTable idt = new IndexTable(array);
        List<Double> test = new DoubleList();
        for(double d : array)
            test.add(d);
        
        idt.apply(test);
        for(int i = 0; i < test.size()-1; i++)
            assertTrue(test.get(i) <= test.get(i+1));
    }
    
    @Test
    public void testSwap()
    {
        System.out.println("swap");
        int i = 0;
        int j = 1;
        IndexTable idx = new IndexTable(array);
        double di = array[idx.index(i)];
        double dj = array[idx.index(j)];
        idx.swap(i, j);
        assertTrue(di == array[idx.index(j)]);
        assertTrue(dj == array[idx.index(i)]);
        assertTrue(di != dj);
        idx.swap(j, i);
        assertTrue(di != array[idx.index(j)]);
        assertTrue(dj != array[idx.index(i)]);
        assertTrue(di == array[idx.index(i)]);
        assertTrue(dj == array[idx.index(j)]);
        
    }
}
