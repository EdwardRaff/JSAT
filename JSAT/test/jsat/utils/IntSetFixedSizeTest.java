
package jsat.utils;

import java.util.*;
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
public class IntSetFixedSizeTest
{
    
    public IntSetFixedSizeTest()
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
     * Test of add method, of class IntSetFixedSize.
     */
    @Test
    public void testAdd_Integer()
    {
        System.out.println("add");
        IntSetFixedSize set = new IntSetFixedSize(5);
        assertTrue(set.add(new Integer(1)));
        assertTrue(set.add(new Integer(2)));
        assertFalse(set.add( new Integer(1) ));
        assertTrue(set.add( new Integer(3) ));

        for (int badVal : Arrays.asList(-1, 5, 10, -23))
            try
            {
                set.add( new Integer(badVal));
                fail("Should not have been added");
            }
            catch (Exception ex)
            {
            
            }
    }

    /**
     * Test of add method, of class IntSetFixedSize.
     */
    @Test
    public void testAdd_int()
    {
        System.out.println("add");
        IntSetFixedSize set = new IntSetFixedSize(5);
        assertTrue(set.add(1));
        assertTrue(set.add(2));
        assertFalse(set.add(1));
        assertTrue(set.add(3));

        for (int badVal : Arrays.asList(-1, 5, 10, -23))
            try
            {
                set.add(badVal);
                fail("Should not have been added");
            }
            catch (Exception ex)
            {
            }
    }

    /**
     * Test of contains method, of class IntSetFixedSize.
     */
    @Test
    public void testContains_Object()
    {
        System.out.println("contains");
        IntSetFixedSize instance = new IntSetFixedSize(100);
        List<Integer> intList = new IntList();
        ListUtils.addRange(intList, 0, 100, 1);
        Collections.shuffle(intList);
        
        instance.addAll(intList.subList(0, 50));
        for(int i : intList.subList(0, 50))
            assertTrue(instance.contains(i));
        for(int i : intList.subList(50, 100))
            assertFalse(instance.contains(i));
        
    }

    /**
     * Test of contains method, of class IntSetFixedSize.
     */
    @Test
    public void testContains_int()
    {
        System.out.println("contains");
        IntSetFixedSize instance = new IntSetFixedSize(100);
        List<Integer> intList = new IntList();
        ListUtils.addRange(intList, 0, 100, 1);
        Collections.shuffle(intList);
        
        instance.addAll(intList.subList(0, 50));
        for(int i : intList.subList(0, 50))
            assertTrue(instance.contains(i));
        for(int i : intList.subList(50, 100))
            assertFalse(instance.contains(i));
    }

    /**
     * Test of iterator method, of class IntSetFixedSize.
     */
    @Test
    public void testIterator()
    {
        System.out.println("iterator");
        IntSetFixedSize set = new IntSetFixedSize(10);
        set.add(5);
        set.add(3);
        set.add(4);
        set.add(1);
        set.add(2);
        Set<Integer> copySet = new IntSet(set);
        int prev = Integer.MIN_VALUE;
        Iterator<Integer> iter = set.iterator();
        int count = 0;
        while(iter.hasNext())
        {
            int val = iter.next();
            copySet.remove(val);
            count++;
        }
        assertEquals(5, set.size());
        assertEquals(5, count);
        assertEquals(0, copySet.size());
        
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
     * Test of size method, of class IntSetFixedSize.
     */
    @Test
    public void testSize()
    {
        System.out.println("size");
        IntSetFixedSize set = new IntSetFixedSize(10);
        assertEquals(0, set.size());
        set.add(1);
        assertEquals(1, set.size());
        set.add(1);
        set.add(2);
        assertEquals(2, set.size());
        set.add(5);
        set.add(7);
        set.add(2);
        assertEquals(4, set.size());
    }
}
