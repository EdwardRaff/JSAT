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
package jsat.utils;

import java.util.*;
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
 * @author Edward Raff
 */
public class FibHeapTest
{
    
    public FibHeapTest()
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
    public void testDecreaseKey()
    {
        System.out.println("decreaseKey");
        FibHeap<Character> h1 = new FibHeap<Character>();
        List<FibHeap.FibNode<Character>> nodes = new ArrayList<FibHeap.FibNode<Character>>();
        
        List<Character> expectedOrder = new ArrayList<Character>();
        for(char c = 'A'; c < 'z'; c++)
        {
            nodes.add(h1.insert(c, c));
            expectedOrder.add(0, c);
        }
        Collections.shuffle(nodes);
        for(FibHeap.FibNode<Character> node : nodes)
            h1.decreaseKey(node, -node.value);
        
        assertEquals(expectedOrder.size(), h1.size());
        int pos = 0;
        while(h1.size() > 0)
        {
            FibHeap.FibNode<Character> min = h1.removeMin();
            assertEquals(expectedOrder.get(pos), min.value);
            pos++;
        }
    }
    /**
     * Test of union method, of class FibHeap.
     */
    @Test
    public void testUnion()
    {
        System.out.println("union");
        FibHeap<String> h1 = new FibHeap<String>();
        FibHeap<String> h2 = new FibHeap<String>();
        
        double value = 0.0;
        int counter= 0;
        List<String> added = new ArrayList<String>();
        for(char c = 'A'; c <= 'Z'; c++)
        {
            counter++;
            String s = Character.toString(c);
            if(c % 2 == 0)
                h1.insert(s, value--);
            else
                h2.insert(s, value--);
            added.add(0, s);
            
//            if(c % 2 == 0)
//                h1.insert(s, value++);
//            else
//                h2.insert(s, value++);
//            added.add(s);
            
        }
        
        FibHeap<String> heap = FibHeap.union(h1, h2);
        assertEquals(counter, heap.size());
        
        counter = 0;
        while(heap.size() > 0)
        {
            String s = heap.removeMin().value;
            assertEquals(added.get(counter), s);
            counter++;
        }
        
    }

    /**
     * Test of getMinKey method, of class FibHeap.
     */
    @Test
    public void testGetMinKey()
    {
        System.out.println("getMinKey");
        FibHeap<String> heap = new FibHeap<String>();
        
        FibHeap.FibNode<String> min;
        
        assertEquals(0, heap.size());
        heap.insert("A", 1.0);
        heap.insert("B", 0.5);
        heap.insert("C", 2.0);
        assertEquals(3, heap.size());
        assertEquals("B", heap.getMinValue());
        
        min = heap.removeMin();
        assertEquals("B", min.value);
        assertEquals(2, heap.size());
        assertEquals("A", heap.getMinValue());
        
        min = heap.removeMin();
        assertEquals("A", min.value);
        assertEquals(1, heap.size());
        assertEquals("C", heap.getMinValue());
        
        min = heap.removeMin();
        assertEquals("C", min.value);
        assertEquals(0, heap.size());
        
    }
    
    
    @Test
    public void testRandom()
    {
        System.out.println("testRandp,");
        FibHeap<Long> heap = new FibHeap<Long>();
        
        SortedMap<Long, Double> map = new TreeMap<Long, Double>();
        Map<Long, FibHeap.FibNode<Long>> heapNodes = new HashMap<Long, FibHeap.FibNode<Long>>();
        
        Random rand = RandomUtil.getRandom();
        
        for(int trials = 0; trials < 10; trials++)
        for(int maxSize = 1; maxSize < 2000; maxSize*=2)
        {
            while(map.size() < maxSize)
            {
                
                if(map.size() > 0 && rand.nextDouble() < 0.1)
                {//ocasionally remove the min eliment
                    long entry = map.firstKey();
                    double value = map.get(entry);
                    for( Map.Entry<Long, Double> mapEntry : map.entrySet())
                        if(mapEntry.getValue() < value)
                        {
                            entry = mapEntry.getKey();
                            value = mapEntry.getValue();
                        }
                    map.remove(entry);
                    
                    FibHeap.FibNode<Long> min = heap.removeMin();
                    heapNodes.remove(entry);
                    
                    assertEquals(entry, min.value.longValue());
                    assertEquals(value, min.key, 0.0);

                }
                else if(map.size() > 0 && rand.nextDouble() < 0.4)
                {//lest decrease a random key's value
                    long partitioner = rand.nextLong();
                    SortedMap<Long, Double> subMap = map.tailMap(partitioner);
                    
                    long valToDecrease;
                    if(subMap.isEmpty())
                    {
                        subMap = map.headMap(partitioner);
                        valToDecrease = map.lastKey();
                    }
                    else
                    {
                        valToDecrease = map.firstKey();
                    }
                    
                    double newVal = map.get(valToDecrease)/2;
                    map.put(valToDecrease, newVal);
                    heap.decreaseKey(heapNodes.get(valToDecrease), newVal);
                    
                }
                //then add something
                long entry = rand.nextLong();
                double value = rand.nextDouble();
                while(map.containsKey(entry))
                    entry = rand.nextLong();
                map.put(entry, value);
                heapNodes.put(entry, heap.insert(entry, value));
                
            }
            
            //its full, now lets remove everything!
            while(map.size() > 0)
            {
                long entry = map.firstKey();
                double value = map.get(entry);
                for (Map.Entry<Long, Double> mapEntry : map.entrySet())
                    if (mapEntry.getValue() < value)
                    {
                        entry = mapEntry.getKey();
                        value = mapEntry.getValue();
                    }
                map.remove(entry);

                FibHeap.FibNode<Long> min = heap.removeMin();
                heapNodes.remove(entry);

                assertEquals(entry, min.value.longValue());
                assertEquals(value, min.key, 0.0);

            }
        }
        
    }
}
