/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import java.util.List;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class ListUtilsTest
{
    
    public ListUtilsTest()
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

    /**
     * Test of splitList method, of class ListUtils.
     */
    @Test
    public void testSplitList()
    {
        System.out.println("splitList");
        List<Integer> sourceList = new IntList(500);
        for(int i = 0; i < 500; i++)
            sourceList.add(i);
        List<List<Integer>> ll1 = ListUtils.splitList(sourceList, 5);
        assertEquals(5, ll1.size());
        
        for(int i = 0; i < 5; i++)
        {
            List<Integer> l = ll1.get(i);
            assertEquals(100, l.size());
            for(int j = 0; j < l.size(); j++)
                assertEquals( i*100+j, l.get(j).intValue());//intValue called b/c it becomes ambigous to the compiler without it
        }
        
        
        ll1 = ListUtils.splitList(sourceList, 7);//Non divisible amount
        assertEquals(7, ll1.size());
        int pos = 0;
        for(List<Integer> l : ll1)
        {
            assertTrue("List should have had only 71 or 72 values", l.size() == 72 || l.size() == 71 );
            for(int j = 0; j < l.size(); j++)
            {
                assertEquals(pos+j, l.get(j).intValue());
            }
            pos += l.size();
        }
        assertEquals(500, pos);
    }
    
    @Test
    public void testSwap()
    {
        System.out.println("swap");
        List<Long> test = new LongList();
        test.add(1L);
        test.add(2L);
        ListUtils.swap(test, 0, 1);
        assertEquals(2, test.get(0).longValue());
        assertEquals(1, test.get(1).longValue());
        
        ListUtils.swap(test, 0, 1);
        assertEquals(1, test.get(0).longValue());
        assertEquals(2, test.get(1).longValue());
        
        ListUtils.swap(test, 0, 0);
        assertEquals(1, test.get(0).longValue());
        assertEquals(2, test.get(1).longValue());
    }
}
