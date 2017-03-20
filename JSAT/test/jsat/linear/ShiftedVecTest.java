

package jsat.linear;

import java.util.Iterator;
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
public class ShiftedVecTest
{
    private Vec a_base;
    private Vec b_base;
    
    private double shift_a;
    private double shift_b;
    
    private Vec a_dense;
    private Vec a_sparse;
    private Vec b_dense;
    private Vec b_sparse;
    
    private Vec rand_x;
    private Vec rand_y;
    
    public ShiftedVecTest()
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
        a_base = new SparseVector(12, 6);
        a_base.set(0, 1.0);
        a_base.set(1, 2.0);
        a_base.set(3, -2.0);
        a_base.set(4, 5.0);
        a_base.set(8, -3.0);
        a_base.set(11, 1.0);
        
        b_base = new SparseVector(12, 6);
        b_base.set(0, 1.0);
        b_base.set(1, -2.0);
        b_base.set(3, -3.0);
        b_base.set(4, 4.0);
        b_base.set(7, -2.0);
        b_base.set(10, 1.0);
        
        shift_a = -1;
        shift_b = 2;
        
        a_dense = new DenseVector(a_base);
        a_sparse = new SparseVector(a_base);
        
        a_dense.mutableAdd(shift_a);
        a_sparse.mutableAdd(shift_a);
        
        b_dense = new DenseVector(b_base);
        b_sparse = new SparseVector(b_base);
        
        b_dense.mutableAdd(shift_b);
        b_sparse.mutableAdd(shift_b);
        
        Random rand = RandomUtil.getRandom();
        rand_x = new DenseVector(a_base.length());
        rand_y = new DenseVector(a_base.length());
        
        for(int i =0 ; i < rand_x.length(); i++)
        {
            rand_x.set(i, Math.round(rand.nextDouble()*10));
            rand_y.set(i, Math.round(rand.nextDouble()*10));
        }
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of setShift method, of class ShiftedVec.
     */
    @Test
    public void testSetShift()
    {
        System.out.println("setShift");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        
        assertEquals(shift_a, a.getShift(), 0.0);
        
        a.setShift(shift_b);
        
        assertEquals(shift_b, a.getShift(), 0.0);
    }

    /**
     * Test of embedShift method, of class ShiftedVec.
     */
    @Test
    public void testEmbedShift()
    {
        System.out.println("embedShift");
        
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        
        a.embedShift();
        
        assertEquals(0.0, a.getShift(), 0.0);
        
        assertTrue(a.getBase().equals(a_dense, 1e-10));
    }

    /**
     * Test of length method, of class ShiftedVec.
     */
    @Test
    public void testLength()
    {
        System.out.println("length");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_base.length(), a.length());
    }

    /**
     * Test of get method, of class ShiftedVec.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        
        for(int i =0 ; i < a.length(); i++)
        {
            assertEquals(a_dense.get(i), a.get(i), 1e-10);
            assertEquals(b_dense.get(i), b.get(i), 1e-10);
        }
    }

    /**
     * Test of set method, of class ShiftedVec.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        
        Random rand = RandomUtil.getRandom();
        for(int round = 0; round < 100; round++)
        {
            int indx = rand.nextInt(a.length());
            double tmp = rand.nextDouble();
            a.set(indx, tmp);
            assertEquals(tmp, a.get(indx), 1e-10);
            assertEquals(tmp-shift_a, a_base.get(indx), 1e-10);
        }
    }

    /**
     * Test of increment method, of class ShiftedVec.
     */
    @Test
    public void testIncrement()
    {
        System.out.println("increment");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        
        for(int i =0 ; i < a.length(); i++)
        {
            a.increment(i, (i+1));
            a_dense.increment(i, (i+1));
            assertEquals(a_dense.get(i), a.get(i), 1e-10);
        }
    }

    /**
     * Test of mutableAdd method, of class ShiftedVec.
     */
    @Test
    public void testMutableAdd_Vec()
    {
        System.out.println("mutableAdd");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutableAdd(rand_x);
        a_dense.mutableAdd(rand_x);
        
        assertTrue(a.equals(a_dense, 1e-10));
        
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        b.mutableAdd(a);
        b_dense.mutableAdd(a_dense);
        
        
        assertTrue(b.equals(b_dense, 1e-10));
    }

    /**
     * Test of mutableAdd method, of class ShiftedVec.
     */
    @Test
    public void testMutableAdd_double()
    {
        System.out.println("mutableAdd");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutableAdd(10);
        a_dense.mutableAdd(10);
        
        assertTrue(a.equals(a_dense, 1e-10));
        
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        b.mutableAdd(2);
        b_dense.mutableAdd(2);
        
        assertTrue(b.equals(b_dense, 1e-10));
    }

    /**
     * Test of mutableAdd method, of class ShiftedVec.
     */
    @Test
    public void testMutableAdd_double_Vec()
    {
        System.out.println("mutableAdd");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutableAdd(10, rand_x);
        a_dense.mutableAdd(10, rand_x);
        
        assertTrue(a.equals(a_dense, 1e-10));
        
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        b.mutableAdd(-3, a);
        b_dense.mutableAdd(-3, a_dense);
        
        
        assertTrue(b.equals(b_dense, 1e-10));
    }

    /**
     * Test of mutableDivide method, of class ShiftedVec.
     */
    @Test
    public void testMutableDivide()
    {
        System.out.println("mutableDivide");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutableDivide(10);
        a_dense.mutableDivide(10);
        
        assertTrue(a.equals(a_dense, 1e-10));
    }

    /**
     * Test of mutableMultiply method, of class ShiftedVec.
     */
    @Test
    public void testMutableMultiply()
    {
        System.out.println("mutableMultiply");
        
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutableMultiply(10);
        a_dense.mutableMultiply(10);
        
        assertTrue(a.equals(a_dense, 1e-10));
        
        
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        b.mutableMultiply(0);
        b_dense.mutableMultiply(0);
        
        assertTrue(b.equals(b_dense, 1e-10));
        
        assertEquals(0.0, b.getShift(), 0.0);
    }

    /**
     * Test of mutablePairwiseDivide method, of class ShiftedVec.
     */
    @Test
    public void testMutablePairwiseDivide()
    {
        System.out.println("mutablePairwiseDivide");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutablePairwiseDivide(rand_x.add(1));
        a_dense.mutablePairwiseDivide(rand_x.add(1));
        
        assertTrue(a.equals(a_dense, 1e-10));
    }

    /**
     * Test of mutablePairwiseMultiply method, of class ShiftedVec.
     */
    @Test
    public void testMutablePairwiseMultiply()
    {
        System.out.println("mutablePairwiseMultiply");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.mutablePairwiseDivide(rand_x);
        a_dense.mutablePairwiseDivide(rand_x);
        
        assertTrue(a.equals(a_dense, 1e-10));
    }

    /**
     * Test of dot method, of class ShiftedVec.
     */
    @Test
    public void testDot()
    {
        System.out.println("dot");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        
        assertEquals(a_dense.dot(rand_x), a.dot(rand_x), 1e-10);
        assertEquals(a_dense.dot(rand_x), rand_x.dot(a), 1e-10);
        
        assertEquals(a_dense.dot(b_dense), a.dot(b), 1e-10);//shift on shift
        assertEquals(a_dense.dot(b_dense), a.dot(b_dense), 1e-10);//shift on dense
        assertEquals(a_dense.dot(b_dense), a_dense.dot(b), 1e-10);//dense on shift
        assertEquals(a_dense.dot(b_dense), a.dot(b_sparse), 1e-10);//shift on sparse
        assertEquals(a_dense.dot(b_dense), a_sparse.dot(b), 1e-10);//sparse on shift
    }

    /**
     * Test of zeroOut method, of class ShiftedVec.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        a.zeroOut();
        
        for(int i = 0; i < a.length(); i++)
            assertEquals(0.0, a.get(i), 0.0);
        assertEquals(0.0, a.getShift(), 0.0);
    }

    /**
     * Test of pNorm method, of class ShiftedVec.
     */
    @Test
    public void testPNorm()
    {
        System.out.println("pNorm");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        double[] ps = {0.5, 1, 2, 2.5, 3, 10 };
        
        for(double p : ps)
            assertEquals(a_dense.pNorm(p), a.pNorm(p), 1e-10);
    }

    /**
     * Test of mean method, of class ShiftedVec.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.mean(), a.mean(), 1e-10);
    }

    /**
     * Test of variance method, of class ShiftedVec.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.variance(), a.variance(), 1e-10);
    }

    /**
     * Test of standardDeviation method, of class ShiftedVec.
     */
    @Test
    public void testStandardDeviation()
    {
        System.out.println("standardDeviation");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.standardDeviation(), a.standardDeviation(), 1e-10);
    }

    /**
     * Test of kurtosis method, of class ShiftedVec.
     */
    @Test
    public void testKurtosis()
    {
        System.out.println("kurtosis");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.kurtosis(), a.kurtosis(), 1e-10);
    }

    /**
     * Test of max method, of class ShiftedVec.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.max(), a.max(), 1e-10);
    }

    /**
     * Test of min method, of class ShiftedVec.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.min(), a.min(), 1e-10);
    }

    /**
     * Test of median method, of class ShiftedVec.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        assertEquals(a_dense.median(), a.median(), 1e-10);
    }

    /**
     * Test of getNonZeroIterator method, of class ShiftedVec.
     */
    @Test
    public void testGetNonZeroIterator()
    {
        System.out.println("getNonZeroIterator");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        ShiftedVec b = new ShiftedVec(b_base, shift_b);
        ShiftedVec c = new ShiftedVec(new ConstantVector(1.0, 12), -1);
        
        for(int start = 0; start < a.length(); start++)
        {
            Iterator<IndexValue> expIter = a_sparse.getNonZeroIterator(start);
            Iterator<IndexValue> actIter = a.getNonZeroIterator(start);
            
            assertTrue(expIter.hasNext() == actIter.hasNext());
            
            while(expIter.hasNext())
            {
                IndexValue expIV = expIter.next();
                IndexValue actIV = actIter.next();
                
                assertEquals(expIV.getIndex(), actIV.getIndex());
                assertEquals(expIV.getValue(), actIV.getValue(), 1e-10);
            }
            
            assertTrue(expIter.hasNext() == actIter.hasNext());
            
            //2nd pair
            expIter = b_sparse.getNonZeroIterator(start);
            actIter = b.getNonZeroIterator(start);
            
            assertTrue(expIter.hasNext() == actIter.hasNext());
            
            while(expIter.hasNext())
            {
                IndexValue expIV = expIter.next();
                IndexValue actIV = actIter.next();
                
                assertEquals(expIV.getIndex(), actIV.getIndex());
                assertEquals(expIV.getValue(), actIV.getValue(), 1e-10);
                
                assertTrue(expIter.hasNext() == actIter.hasNext());
            }
            
            assertTrue(expIter.hasNext() == actIter.hasNext());
            
            
            //3rd pair
            expIter = new SparseVector(c.length()).getNonZeroIterator(start);
            actIter = c.getNonZeroIterator(start);
            
            assertTrue(expIter.hasNext() == actIter.hasNext());
            
            while(expIter.hasNext())
            {
                IndexValue expIV = expIter.next();
                IndexValue actIV = actIter.next();
                
                assertEquals(expIV.getIndex(), actIV.getIndex());
                assertEquals(expIV.getValue(), actIV.getValue(), 1e-10);
                
                assertTrue(expIter.hasNext() == actIter.hasNext());
            }
            
            assertTrue(expIter.hasNext() == actIter.hasNext());
        }
    }

    /**
     * Test of isSparse method, of class ShiftedVec.
     */
    @Test
    public void testIsSparse()
    {
        System.out.println("isSparse");
        assertTrue(new ShiftedVec(a_base, shift_a).isSparse());
        assertFalse(new ShiftedVec(a_dense, shift_a).isSparse());
    }

    /**
     * Test of clone method, of class ShiftedVec.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        ShiftedVec a = new ShiftedVec(a_base, shift_a);
        ShiftedVec a_clone = a.clone();
        assertTrue(a.equals(a_dense, 1e-10));
        assertTrue(a_clone.equals(a_dense, 1e-10));
        
        a.mutableAdd(rand_x);
        
        assertFalse(a.equals(a_dense, 1e-10));
        assertTrue(a_clone.equals(a_dense, 1e-10));
    }
    
}
