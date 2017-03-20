package jsat.linear;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
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
public class ConcatenatedVecTest
{
    ConcatenatedVec cvec;
    DenseVector dvec;
    
    public ConcatenatedVecTest()
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
        List<Vec> vecs = new ArrayList<Vec>();
        vecs.add(DenseVector.toDenseVec(0, 1, 2));
        vecs.add(DenseVector.toDenseVec(3, 4, 5));
        vecs.add(DenseVector.toDenseVec(6, 7, 8));
        vecs.add(DenseVector.toDenseVec(9, 10, 11));
        vecs.add(DenseVector.toDenseVec(0, 0, 14));
        
        dvec = DenseVector.toDenseVec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 14);
        cvec = new ConcatenatedVec(vecs);
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of length method, of class ConcatenatedVec.
     */
    @Test
    public void testLength()
    {
        System.out.println("length");
        assertEquals(dvec.length(), cvec.length());
    }

    /**
     * Test of get method, of class ConcatenatedVec.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        for(int i = 0; i < dvec.length(); i++)
            assertEquals(dvec.get(i), cvec.get(i), 0.0);
        
        try
        {
            cvec.get(cvec.length());
            fail("Index out of bounds should have occured");
        }
        catch( IndexOutOfBoundsException ex)
        {
            //good, that was supposed to happen
        }
    }

    /**
     * Test of set method, of class ConcatenatedVec.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        Random rand = RandomUtil.getRandom();
        for(int i = 0; i < dvec.length(); i++)
        {
            double nv = rand.nextDouble();
            dvec.set(i, nv);
            cvec.set(i, nv);
        }
        
        for(int i = 0; i < dvec.length(); i++)
            assertEquals(dvec.get(i), cvec.get(i), 0.0);
    }
    
    @Test
    public void testMutableAdd()
    {
        System.out.println("mutableAdd");
        cvec.mutableAdd(-1, dvec);
        for(int i = 0; i < dvec.length(); i++)
            assertEquals(0.0, cvec.get(i), 0.0);
    }
    
    @Test
    public void testGetNonZeroIterator()
    {
        System.out.println("getNonZeroIterator");
        for(int i = 0; i < dvec.length(); i++)
        {
            Iterator<IndexValue> diter = dvec.getNonZeroIterator(i);
            Iterator<IndexValue> citer = cvec.getNonZeroIterator(i);
            
            assertTrue(diter.hasNext() == citer.hasNext());
            
            while(diter.hasNext())
            {
                IndexValue dIV = diter.next();
                IndexValue cIV = citer.next();
                
                assertEquals(dIV.getIndex(), cIV.getIndex());
                assertEquals(dIV.getValue(), cIV.getValue(), 0.0);
                assertTrue(diter.hasNext() == citer.hasNext());
            }
            
            assertTrue(diter.hasNext() == citer.hasNext());
        }
    }

    /**
     * Test of isSparse method, of class ConcatenatedVec.
     */
    @Test
    public void testIsSparse()
    {
        System.out.println("isSparse");
        assertFalse(cvec.isSparse());
    }

    /**
     * Test of clone method, of class ConcatenatedVec.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        ConcatenatedVec cloned = cvec.clone();
        cvec.mutableAdd(-1, dvec);
        for(int i = 0; i < dvec.length(); i++)
            assertEquals(0.0, cvec.get(i), 0.0);
        
        for(int i = 0; i < dvec.length(); i++)
            assertEquals(dvec.get(i), cloned.get(i), 0.0);
    }
}
