package jsat.linear;

import java.util.Iterator;
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
public class Poly2VecTest
{
    
    Vec baseVec;
    Vec denseBase;
    Vec truePolyVec;
    Vec truePolyDense;
    int[] x, y;
    
    public Poly2VecTest()
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
        baseVec = new DenseVector(new double[]{2.0, 0.0, 3.0, 5.0, 0.0, 0.0, 7.0, 0.0});
        denseBase = new DenseVector(new double[]{2.0, 3.0, 5.0, 7.0});
        truePolyVec = new DenseVector(45);
        truePolyDense = new DenseVector(15);
        truePolyDense.set(0, 1.0);
        truePolyVec.set(0, 1.0);
        for(int i = 0; i < baseVec.length(); i++)
            truePolyVec.set(i+1, baseVec.get(i));
        for(int i = 0; i < denseBase.length(); i++)
            truePolyDense.set(i+1, denseBase.get(i));
        int offSet = baseVec.length()+1;
        int pos = 0;
        x = new int[truePolyVec.length()];
        y = new int[truePolyVec.length()];
        for(int i = 0; i < baseVec.length(); i++)
            for(int j = i; j < baseVec.length(); j++)
            {
                x[pos] = i;
                y[pos] = j;
                truePolyVec.set(offSet + (pos++), baseVec.get(i)*baseVec.get(j));
            }
        
        offSet = denseBase.length()+1;
        pos = 0;
        for(int i = 0; i < denseBase.length(); i++)
            for(int j = i; j < denseBase.length(); j++)
            {
                truePolyDense.set(offSet + (pos++), denseBase.get(i)*denseBase.get(j));
            }
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of length method, of class Poly2Vec.
     */
    @Test
    public void testLength()
    {
        System.out.println("length");
        Poly2Vec polyDense = new Poly2Vec(denseBase);
        assertEquals(truePolyDense.length(), polyDense.length());
        
        Poly2Vec polyVec = new Poly2Vec(baseVec);
        assertEquals(truePolyVec.length(), polyVec.length());
    }

    /**
     * Test of nnz method, of class Poly2Vec.
     */
    @Test
    public void testNnz()
    {
        System.out.println("nnz");
        Poly2Vec polyDense = new Poly2Vec(denseBase);
        assertEquals(truePolyDense.nnz(), polyDense.nnz());
        
        Poly2Vec polyVec = new Poly2Vec(baseVec);
        assertEquals(truePolyVec.nnz(), polyVec.nnz());
    }

    /**
     * Test of get method, of class Poly2Vec.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        Poly2Vec polyDense = new Poly2Vec(denseBase);
        for(int i = 0; i < truePolyDense.length(); i++)
            assertEquals(truePolyDense.get(i), polyDense.get(i), 0.0);
        
        Poly2Vec polyVec = new Poly2Vec(baseVec);
        for(int i = 0; i < truePolyVec.length(); i++)
            assertEquals(truePolyVec.get(i), polyVec.get(i), 0.0);
        try
        {
            polyVec.get(-1);
            fail("Should not be able to access Index");
        }
        catch(IndexOutOfBoundsException ex)
        {
            //good!
        }
        
        try
        {
            polyVec.get(polyVec.length());
            fail("Should not be able to access Index");
        }
        catch(IndexOutOfBoundsException ex)
        {
            //good!
        }
    }

    /**
     * Test of set method, of class Poly2Vec.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        Poly2Vec polyVec = new Poly2Vec(baseVec);
        try
        {
            polyVec.set(2, Double.MAX_VALUE);
            fail("Should not be able to alter poly vec wrappers");
        }
        catch(Exception ex)
        {
            //good!
        }
    }

    /**
     * Test of isSparse method, of class Poly2Vec.
     */
    @Test
    public void testIsSparse()
    {
        System.out.println("isSparse");
        Poly2Vec polyVec = new Poly2Vec(baseVec);
        assertFalse(polyVec.isSparse());
        polyVec = new Poly2Vec(new SparseVector(baseVec));
        assertTrue(polyVec.isSparse());
    }

    /**
     * Test of clone method, of class Poly2Vec.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        
        assertEquals(truePolyDense, new Poly2Vec(new SparseVector(denseBase)).clone());
        assertEquals(truePolyVec, new Poly2Vec(new SparseVector(baseVec)).clone());
    }

    /**
     * Test of getNonZeroIterator method, of class Poly2Vec.
     */
    @Test
    public void testGetNonZeroIterator()
    {
        System.out.println("getNonZeroIterator");
        Poly2Vec polyDense = new Poly2Vec(denseBase);
        for(int i = 0; i < truePolyDense.length(); i++)
        {
            Iterator<IndexValue> trueIter = truePolyDense.getNonZeroIterator(i);
            Iterator<IndexValue> polyIter = polyDense.getNonZeroIterator(i);
            
            assertTrue(trueIter.hasNext() == polyIter.hasNext());
            
            while(trueIter.hasNext())
            {
                assertTrue(trueIter.hasNext() == polyIter.hasNext());
                IndexValue trueIV = trueIter.next();
                IndexValue polyIV = polyIter.next();
                
                assertEquals(trueIV.getIndex(), polyIV.getIndex());
                assertEquals(trueIV.getValue(), polyIV.getValue(), 0.0);
                
                assertTrue(trueIter.hasNext() == polyIter.hasNext());
            }
            
        }
        
        
        Poly2Vec polyVec = new Poly2Vec(baseVec);
        for(int i = 0; i < truePolyVec.length(); i++)
        {
            Iterator<IndexValue> trueIter = truePolyVec.getNonZeroIterator(i);
            Iterator<IndexValue> polyIter = polyVec.getNonZeroIterator(i);
            
            assertTrue(trueIter.hasNext() == polyIter.hasNext());
            
            while(trueIter.hasNext())
            {
                assertTrue(trueIter.hasNext() == polyIter.hasNext());
                IndexValue trueIV = trueIter.next();
                IndexValue polyIV = polyIter.next();
                
                assertEquals(trueIV.getIndex(), polyIV.getIndex());
                assertEquals(trueIV.getValue(), polyIV.getValue(), 0.0);
                
                assertTrue(trueIter.hasNext() == polyIter.hasNext());
            }
            
        }
    }
}
