/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import java.util.Random;
import org.junit.*;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class IntPriorityQueueTest
{
    
    public IntPriorityQueueTest()
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
    public void testGeneralStanrdardMode()
    {
        IntPriorityQueue ipq = new IntPriorityQueue(8, IntPriorityQueue.Mode.STANDARD);
        runSmallGambit(ipq);
    }
    
    @Test
    public void testGeneralHashdMode()
    {
        IntPriorityQueue ipq = new IntPriorityQueue(8, IntPriorityQueue.Mode.HASH);
        runSmallGambit(ipq);
    }
    
    @Test
    public void testGeneralBoundedMode()
    {
        IntPriorityQueue ipq = new IntPriorityQueue(8, IntPriorityQueue.Mode.BOUNDED);
        runSmallGambit(ipq);
    }

    private void runSmallGambit(IntPriorityQueue ipq)
    {
        ipq.add(2);
        assertEquals(1, ipq.size());
        assertEquals(2L, ipq.peek().intValue());
        assertEquals(1, ipq.size());
        
        assertEquals(2L, ipq.poll().intValue());
        assertEquals(0, ipq.size());
        assertNull(ipq.peek());
        assertNull(ipq.poll());
        
        ipq.add(3);
        ipq.add(1);
        ipq.add(7);
        
        assertEquals(1, ipq.peek().intValue());
        assertEquals(3, ipq.size());
        
        ipq.remove(1);
        ipq.add(2);
        
        assertEquals(2, ipq.peek().intValue());
        assertEquals(3, ipq.size());
        
        ipq.add(1);
        
        assertEquals(1, ipq.peek().intValue());
        assertEquals(4, ipq.size());
        
        assertTrue(ipq.contains(1));
        assertTrue(ipq.contains(2));
        assertTrue(ipq.contains(3));
        assertFalse(ipq.contains(4));
        assertFalse(ipq.contains(5));
        assertTrue(ipq.contains(7));
        
        assertEquals(1, ipq.poll().intValue());
        assertEquals(3, ipq.size());
        assertEquals(2, ipq.poll().intValue());
        assertEquals(2, ipq.size());
        assertEquals(3, ipq.poll().intValue());
        assertEquals(1, ipq.size());
        assertEquals(7, ipq.poll().intValue());
        assertEquals(0, ipq.size());
        assertNull(ipq.poll());
        assertEquals(0, ipq.size());
        
        Random rand = new Random(2);
        for(int i = 0; i < 100; i++)
            ipq.add(rand.nextInt(200));
        
        int prev = -1;
        while(!ipq.isEmpty())
        {
            int pop = ipq.poll();
            assertTrue(prev <= pop);
            prev = pop;
        }
    }
}
