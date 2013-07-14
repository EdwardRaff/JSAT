package jsat.linear;

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
public class SubVectorTest
{
    private Vec x;
    private Vec y;
    
    public SubVectorTest()
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
        x = new SparseVector(20);
        x.set(3, 1.0);
        x.set(6, 1.0);
        x.set(7, 1.0);
        x.set(12, 1.0);
        x.set(15, 1.0);
        x.set(19, 1.0);
        
        y = new DenseVector(x);
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of length method, of class SubVector.
     */
    @Test
    public void testLength()
    {
        System.out.println("length");
        
        SubVector subX = new SubVector(2, 7, x);
        assertEquals(7, subX.length());
        
        SubVector subY = new SubVector(2, 7, y);
        assertEquals(7, subY.length());
        
    }

    /**
     * Test of get method, of class SubVector.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        SubVector subX = new SubVector(2, 7, x);
        assertEquals(0.0, subX.get(2 - 2), 1e-20);
        assertEquals(1.0, subX.get(3 - 2), 1e-20);
        assertEquals(0.0, subX.get(4 - 2), 1e-20);
        assertEquals(0.0, subX.get(5 - 2), 1e-20);
        assertEquals(1.0, subX.get(6 - 2), 1e-20);
        assertEquals(1.0, subX.get(7 - 2), 1e-20);
        assertEquals(0.0, subX.get(8 - 2), 1e-20);
        
        try
        {
            assertEquals(0.0, subX.get(9 - 2), 1e-20);
            fail("Should not have been able to access value");
        }
        catch (Exception ex)
        {
        }


        SubVector subY = new SubVector(2, 7, y);
        assertEquals(0.0, subY.get(2 - 2), 1e-20);
        assertEquals(1.0, subY.get(3 - 2), 1e-20);
        assertEquals(0.0, subY.get(4 - 2), 1e-20);
        assertEquals(0.0, subY.get(5 - 2), 1e-20);
        assertEquals(1.0, subY.get(6 - 2), 1e-20);
        assertEquals(1.0, subY.get(7 - 2), 1e-20);
        assertEquals(0.0, subY.get(8 - 2), 1e-20);
        try
        {
            assertEquals(0.0, subY.get(9 - 2), 1e-20);
            fail("Should not have been able to access value");
        }
        catch (Exception ex)
        {
        }
    }

    /**
     * Test of set method, of class SubVector.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        SubVector subX = new SubVector(2, 7, x);
        subX.set(2 - 2, -1.0);
        subX.set(3 - 2, -1.0);
        subX.set(4 - 2, -1.0);
        
        assertEquals(-1.0, subX.get(2 - 2), 1e-20);
        assertEquals(-1.0, subX.get(3 - 2), 1e-20);
        assertEquals(-1.0, subX.get(4 - 2), 1e-20);
        
        assertEquals(-1.0, x.get(2), 1e-20);
        assertEquals(-1.0, x.get(3), 1e-20);
        assertEquals(-1.0, x.get(4), 1e-20);
        
        try
        {
            subX.set(9-2, -1.0);
            fail("Should not have been able to access value");
        }
        catch (Exception ex)
        {
        }


        SubVector subY = new SubVector(2, 7, y);
        subY.set(2 - 2, -1.0);
        subY.set(3 - 2, -1.0);
        subY.set(4 - 2, -1.0);
        
        assertEquals(-1.0, subY.get(2 - 2), 1e-20);
        assertEquals(-1.0, subY.get(3 - 2), 1e-20);
        assertEquals(-1.0, subY.get(4 - 2), 1e-20);
        
        assertEquals(-1.0, y.get(2), 1e-20);
        assertEquals(-1.0, y.get(3), 1e-20);
        assertEquals(-1.0, y.get(4), 1e-20);
        
        try
        {
            subY.set(9-2, -1.0);
            fail("Should not have been able to access value");
        }
        catch (Exception ex)
        {
        }
    }

    /**
     * Test of isSparse method, of class SubVector.
     */
    @Test
    public void testIsSparse()
    {
        System.out.println("isSparse");
        SubVector subX = new SubVector(2, 7, x);
        assertTrue(subX.isSparse());
        
        SubVector subY = new SubVector(2, 7, y);
        assertFalse(subY.isSparse());
    }

    /**
     * Test of getNonZeroIterator method, of class SubVector.
     */
    @Test
    public void testGetNonZeroIterator()
    {
        System.out.println("getNonZeroIterator");
        int firstSeen = -1;
        int lastSeen = -1;
        
        SubVector subX = new SubVector(2, 13, x);
        
        firstSeen = subX.getNonZeroIterator(2).next().getIndex();
        for(IndexValue iv : subX)
            lastSeen = iv.getIndex();
        assertEquals(6-2, firstSeen);
        assertEquals(12-2, lastSeen);
        
        firstSeen = -1;
        lastSeen = -1;
        SubVector subY = new SubVector(2, 13, y);
        
        firstSeen = subY.getNonZeroIterator(2).next().getIndex();
        for(IndexValue iv : subY)
            lastSeen = iv.getIndex();
        assertEquals(6-2, firstSeen);
        assertEquals(12-2, lastSeen);
    }

    /**
     * Test of clone method, of class SubVector.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        SubVector subX = new SubVector(2, 7, x);
        Vec subXClone = subX.clone();
        
        assertTrue(subXClone.equals(subX));
        assertFalse(subX == subXClone);
        
        SubVector subY = new SubVector(2, 7, y);
        Vec subYClone = subY.clone();
        
        assertTrue(subYClone.equals(subY));
        assertFalse(subY == subYClone);
    }
}
