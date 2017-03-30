/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import java.util.Iterator;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import jsat.utils.random.RandomUtil;
import org.junit.*;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class IntSortedSetTest
{
    
    public IntSortedSetTest()
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

    /**
     * Test of add method, of class IntSet.
     */
    @Test
    public void testAdd()
    {
        System.out.println("add");
        IntSortedSet set = new IntSortedSet();
        assertFalse(set.add(null));
        assertTrue(set.add(1));
        assertTrue(set.add(2));
        assertFalse(set.add(1));
        assertFalse(set.add(null));
        assertTrue(set.add(3));
    }

    /**
     * Test of iterator method, of class IntSet.
     */
    @Test
    public void testIterator()
    {
        System.out.println("iterator");
        IntSortedSet set = new IntSortedSet();
        set.add(5);
        set.add(3);
        set.add(4);
        set.add(1);
        set.add(2);
        int prev = Integer.MIN_VALUE;
        Iterator<Integer> iter = set.iterator();
        int count = 0;
        while(iter.hasNext())
        {
            int val = iter.next();
            count++;
            assertTrue(prev < val);
            prev = val;
        }
        assertEquals(5, set.size());
        assertEquals(5, count);
        
        //Test removing some elements
        iter = set.iterator();
        while(iter.hasNext())
        {
            int val = iter.next();
            if(val == 2 || val == 4)
                iter.remove();
        }
        assertEquals(3, set.size());
        
        //Make sure the corect values were actually removed
        iter = set.iterator();
        count = 0;
        while(iter.hasNext())
        {
            int val = iter.next();
            assertFalse(val == 2 || val == 4);
            count++;
        }
        assertEquals(3, set.size());
        assertEquals(3, count);
    }

    /**
     * Test of size method, of class IntSet.
     */
    @Test
    public void testSize()
    {
        System.out.println("size");
        IntSortedSet set = new IntSortedSet();
        assertEquals(0, set.size());
        set.add(1);
        assertEquals(1, set.size());
        set.add(1);
        set.add(2);
        assertEquals(2, set.size());
        set.add(5);
        set.add(-4);
        set.add(2);
        assertEquals(4, set.size());
    }
    
    
    @Test
    public void testSubSet()
    {
        System.out.println("subset");

        SortedSet<Integer> groundTruth = new TreeSet<Integer>();
        IntSortedSet testSet = new IntSortedSet();

        for (int i = 1; i < 20; i++)
            for (int j = i * 20; j < (i + 1) * 20; j += i)
            {
                groundTruth.add(j);
                testSet.add(j);
            }

        assertSameContent(groundTruth, testSet);

        Random rand = RandomUtil.getRandom();

        testHeadSet(groundTruth, testSet, rand, 3);
        testTailSet(groundTruth, testSet, rand, 3);
        testSubSet(groundTruth, testSet, rand, 3);
    }

    private void testHeadSet(SortedSet<Integer> groundTruth, SortedSet<Integer> testSet, Random rand, int depth)
    {
        if(groundTruth.isEmpty() || groundTruth.last() - groundTruth.first() <= 0 || groundTruth.last() <= 0)//avoid bad tests
            return;
        int toElement = rand.nextInt(groundTruth.last());
        
        SortedSet<Integer> g_s = groundTruth.headSet(toElement);
        SortedSet<Integer> t_s = testSet.headSet(toElement);
        
        assertSameContent(g_s, t_s);
        for(int i = 0; i < 5; i++)
        {
            int new_val;
            if(toElement <= 0)
                new_val = Math.min(toElement-1, -rand.nextInt(1000));
            else
                new_val = rand.nextInt(toElement);
            g_s.add(new_val);
            t_s.add(new_val);
        }
        assertSameContent(g_s, t_s);
        assertSameContent(groundTruth, testSet);
        
        if(depth-- > 0)
            testHeadSet(g_s, t_s, rand, depth);
        assertSameContent(groundTruth, testSet);
    }
    
    private void testTailSet(SortedSet<Integer> groundTruth, SortedSet<Integer> testSet, Random rand, int depth)
    {
        if(groundTruth.isEmpty() || groundTruth.last() - groundTruth.first() <= 0)//avoid bad tests
            return;
        int fromElement = groundTruth.first() + rand.nextInt(groundTruth.last() - groundTruth.first());
        
        SortedSet<Integer> g_s = groundTruth.tailSet(fromElement);
        SortedSet<Integer> t_s = testSet.tailSet(fromElement);
        
        assertSameContent(g_s, t_s);
        for(int i = 0; i < 5; i++)
        {
            int new_val = fromElement+rand.nextInt(10000);
            g_s.add(new_val);
            t_s.add(new_val);
        }
        assertSameContent(g_s, t_s);
        assertSameContent(groundTruth, testSet);
        
        if(depth-- > 0)
            testTailSet(g_s, t_s, rand, depth);
        assertSameContent(groundTruth, testSet);
    }
    
    private void testSubSet(SortedSet<Integer> groundTruth, SortedSet<Integer> testSet, Random rand, int depth)
    {
        if(groundTruth.isEmpty() || groundTruth.last() - groundTruth.first() <= 0)//avoid bad tests
            return;
        int fromElement = groundTruth.first() + rand.nextInt(groundTruth.last() - groundTruth.first());
        int toElement = fromElement + rand.nextInt(groundTruth.last() - fromElement);
        
        SortedSet<Integer> g_s = groundTruth.subSet(fromElement, toElement);
        SortedSet<Integer> t_s = testSet.subSet(fromElement, toElement);
        
        assertSameContent(g_s, t_s);
        for(int i = 0; i < 5; i++)
        {
            if(fromElement == toElement)
                continue;//we can't add anything
            int new_val = fromElement+rand.nextInt(toElement-fromElement);
            g_s.add(new_val);
            t_s.add(new_val);
        }
        assertSameContent(g_s, t_s);
        assertSameContent(groundTruth, testSet);
        
        if(depth-- > 0)
            testSubSet(g_s, t_s, rand, depth);
        assertSameContent(groundTruth, testSet);
    }
    
    public void assertSameContent(SortedSet<Integer> a, SortedSet<Integer> b)
    {
        assertEquals(a.size(), b.size());
        int counted_a = 0;
        for(int v : a)
        {
            assertTrue(b.contains(v));
            counted_a++;
        }
        assertEquals(a.size(), counted_a);
        
        int counted_b = 0;
        for(int v : b)
        {
            assertTrue(a.contains(v));
            counted_b++;
        }
        assertEquals(b.size(), counted_b);
    }
}
