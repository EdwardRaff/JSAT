
package jsat.utils;

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
public class SimpleListTest
{

    public SimpleListTest()
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
     * Test of get method, of class SimpleList.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        final SimpleList instance = new SimpleList();
        try
        {
            final Object obj = instance.get(0);
            fail("Should not have been able to obtain " + obj);
        }
        catch(final Exception ex)
        {
            
        }
        instance.add("a");
        instance.add("b");
        assertEquals("a", instance.get(0));
        assertEquals("b", instance.get(1));
        
        assertEquals("a", instance.remove(0));
        
        assertEquals("b", instance.get(0));
        try
        {
            final Object obj = instance.get(1);
            fail("Should not have been able to obtain " + obj);
        }
        catch(final Exception ex)
        {
            
        }
        instance.add("c");
        instance.add("d");
        
        assertEquals("b", instance.get(0));
        assertEquals("c", instance.get(1));
        assertEquals("d", instance.get(2));
        try
        {
            final Object obj = instance.get(3);
            fail("Should not have been able to obtain " + obj);
        }
        catch(final Exception ex)
        {
            
        }
    }

    /**
     * Test of size method, of class SimpleList.
     */
    @Test
    public void testSize()
    {
        System.out.println("size");
        final SimpleList instance = new SimpleList();
        assertEquals(0, instance.size());
        instance.add("a");
        instance.add("b");
        assertEquals(2, instance.size());
        assertEquals("a", instance.remove(0));
        assertEquals(1, instance.size());
        instance.add("c");
        instance.add("d");
        assertEquals(3, instance.size());
    }

    @Test
    public void testAdd()
    {
        System.out.println("add");
        final SimpleList<String> instance = new SimpleList<String>();
        instance.add("a");
        instance.add(0, "c");
        instance.add(1, "b");
        assertEquals("c", instance.get(0));
        assertEquals("b", instance.get(1));
        assertEquals("a", instance.get(2));
    }
    
    /**
     * Test of set method, of class SimpleList.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        final SimpleList<String> instance = new SimpleList<String>();
        try
        {
            final Object obj = instance.get(0);
            fail("Should not have been able to obtain " + obj);
        }
        catch(final Exception ex)
        {
            
        }
        instance.add("A");
        instance.add("B");
        instance.set(0, instance.get(0).toLowerCase());
        instance.set(1, instance.get(1).toLowerCase());
        assertEquals("a", instance.get(0));
        assertEquals("b", instance.get(1));
        
        assertEquals("a", instance.remove(0));
        
        assertEquals("b", instance.get(0));
        try
        {
            final Object obj = instance.get(1);
            fail("Should not have been able to obtain " + obj);
        }
        catch(final Exception ex)
        {
            
        }
        instance.add("C");
        instance.add("D");
        
        instance.set(1, instance.get(1).toLowerCase());
        instance.set(2, instance.get(2).toLowerCase());
        assertEquals("b", instance.get(0));
        assertEquals("c", instance.get(1));
        assertEquals("d", instance.get(2));
        try
        {
            final Object obj = instance.get(3);
            fail("Should not have been able to obtain " + obj);
        }
        catch(final Exception ex)
        {
            
        }
    }
}
