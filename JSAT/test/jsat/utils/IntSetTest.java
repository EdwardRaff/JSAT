/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import java.util.Iterator;
import org.junit.*;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class IntSetTest
{
    
    public IntSetTest()
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
        IntSet set = new IntSet();
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
        IntSet set = new IntSet();
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
        IntSet set = new IntSet();
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
}
