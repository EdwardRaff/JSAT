package jsat.linear;

import java.util.Random;
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
public class VecWithNormTest
{
    Vec x, xNrmd;
    Vec a, b, c;
    
    public VecWithNormTest()
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
        Random rand = new Random(546);
        x = DenseVector.random(20, rand);
        xNrmd = new VecWithNorm(x.clone());
        
        a = new SparseVector(x.length());
        b = new SparseVector(x.length());
        c = new SparseVector(x.length());
        
        for(int i = 0; i < 5; i++)
        {
            a.set(rand.nextInt(a.length()), rand.nextGaussian());
            b.set(rand.nextInt(b.length()), rand.nextGaussian());
            c.set(rand.nextInt(c.length()), rand.nextGaussian());
        }
    }
    
    @After
    public void tearDown()
    {
    }


    /**
     * Test of length method, of class VecWithNorm.
     */
    @Test
    public void testLength()
    {
        System.out.println("length");
        assertEquals(x.length(), xNrmd.length());
    }

    /**
     * Test of get method, of class VecWithNorm.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        for(int i = 0; i < x.length(); i++)
            assertEquals(x.get(i), xNrmd.get(i), 0.0);
    }

    /**
     * Test of set method, of class VecWithNorm.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        int index = 16;
        double val = Math.PI;
        x.set(index, val);
        xNrmd.set(index, val);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of clone method, of class VecWithNorm.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        Vec xN = x.clone();
        Vec xNN = xNrmd.clone();
        
        assertEquals(xN, x);
        assertEquals(xNN, xNrmd);
        assertNotSame(xNN, xNrmd);
    }

    /**
     * Test of mutableAdd method, of class VecWithNorm.
     */
    @Test
    public void testMutableAdd_double()
    {
        System.out.println("mutableAdd");
        double val = 3.9;
        
        x.mutableAdd(val);
        xNrmd.mutableAdd(val);
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of mutableAdd method, of class VecWithNorm.
     */
    @Test
    public void testMutableAdd_double_Vec()
    {
        System.out.println("mutableAdd");
        
        double val = 2.555;
        
        x.mutableAdd(val, a);
        xNrmd.mutableAdd(val, a);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
        
        x.mutableAdd(val, b);
        xNrmd.mutableAdd(val, b);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
        
        x.mutableAdd(val, c);        
        xNrmd.mutableAdd(val, c);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of mutablePairwiseMultiply method, of class VecWithNorm.
     */
    @Test
    public void testMutablePairwiseMultiply()
    {
        System.out.println("mutablePairwiseMultiply");
        
        x.mutablePairwiseMultiply(a);
        xNrmd.mutablePairwiseMultiply(a);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
        
        x.mutablePairwiseMultiply(b);
        xNrmd.mutablePairwiseMultiply(b);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
        
        
        x.mutablePairwiseMultiply(c);
        xNrmd.mutablePairwiseMultiply(c);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of mutableMultiply method, of class VecWithNorm.
     */
    @Test
    public void testMutableMultiply()
    {
        System.out.println("mutableMultiply");
        double c = 1.6;
        x.mutableMultiply(c);
        xNrmd.mutableMultiply(c);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of mutableDivide method, of class VecWithNorm.
     */
    @Test
    public void testMutableDivide()
    {
        System.out.println("mutableDivide");
        double c = 2.54;
        
        x.mutableDivide(c);
        xNrmd.mutableDivide(c);
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of zeroOut method, of class VecWithNorm.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        x.zeroOut();
        xNrmd.zeroOut();
        
        assertEquals(x.pNorm(2), xNrmd.pNorm(2), 1e-6);
    }

    /**
     * Test of nnz method, of class VecWithNorm.
     */
    @Test
    public void testNnz()
    {
        System.out.println("nnz");
        
        assertEquals(x.nnz(), xNrmd.nnz());
        x.set(5, 0.0);
        xNrmd.set(5, 0.0);
        assertEquals(x.nnz(), xNrmd.nnz());
    }

    
}
