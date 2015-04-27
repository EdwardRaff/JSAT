package jsat.linear;

import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.math.IndexFunction;
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
public class SparseVectorTest
{
    private SparseVector x;
    private SparseVector y;
    
    public SparseVectorTest()
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
        x.set(0, 2.0);
        x.set(3, 4.0);
        x.set(7, 1.0);
        x.set(10, 2.0);
        x.set(15, 3.0);
        x.set(18, 2.0);
        x.set(19, 5.0);
        
        y = new SparseVector(20);
        y.set(1, 2.0);
        y.set(3, 4.0);
        y.set(8, 1.0);
        y.set(9, 1.0);
        y.set(10, 2.0);
        y.set(15, 3.0);
        y.set(17, 2.0);
        y.set(18, 5.0);
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of length method, of class SparseVector.
     */
    @Test
    public void testLength()
    {
        System.out.println("length");
        assertEquals(20, x.length());
        assertEquals(20, y.length());
    }

    /**
     * Test of setLength method, of class SparseVector.
     */
    @Test
    public void testSetLength()
    {
        System.out.println("setLength");
        assertEquals(20, x.length());
        assertEquals(20, y.length());
        
        y.setLength(25);
        
        assertEquals(20, x.length());
        assertEquals(25, y.length());
        
        try
        {
            y.setLength(5);
            fail("Set length should have been too small");
        }
        catch(Exception ex)
        {
            
        }
    }

    /**
     * Test of nnz method, of class SparseVector.
     */
    @Test
    public void testNnz()
    {
        System.out.println("nnz");
        assertEquals(7, x.nnz());
        assertEquals(8, y.nnz());
    }

    /**
     * Test of increment method, of class SparseVector.
     */
    @Test
    public void testIncrement()
    {
        System.out.println("increment");
        x.increment(0, 2.0);
        x.increment(19, -2.0);
        assertEquals(4.0, x.get(0), 1e-15);
        assertEquals(3.0, x.get(19), 1e-15);
        
        y.increment(0, 2.0);
        y.increment(19, -2.0);
        assertEquals(2.0, y.get(0), 1e-15);
        assertEquals(-2.0, y.get(19), 1e-15);
    }

    /**
     * Test of get method, of class SparseVector.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        assertEquals(2.0, x.get(0), 1e-15);
        assertEquals(4.0, x.get(3), 1e-15);
        assertEquals(1.0, x.get(7), 1e-15);
        assertEquals(2.0, x.get(10), 1e-15);
        assertEquals(3.0, x.get(15), 1e-15);
        assertEquals(2.0, x.get(18), 1e-15);
        assertEquals(5.0, x.get(19), 1e-15);

        assertEquals(0.0, x.get(1), 1e-15);
        assertEquals(0.0, x.get(2), 1e-15);
        assertEquals(0.0, x.get(16), 1e-15);
        

        assertEquals(2.0, y.get(1), 1e-15);
        assertEquals(4.0, y.get(3), 1e-15);
        assertEquals(1.0, y.get(8), 1e-15);
        assertEquals(1.0, y.get(9), 1e-15);
        assertEquals(2.0, y.get(10), 1e-15);
        assertEquals(3.0, y.get(15), 1e-15);
        assertEquals(2.0, y.get(17), 1e-15);
        assertEquals(5.0, y.get(18), 1e-15);

        assertEquals(0.0, y.get(0), 1e-15);
        assertEquals(0.0, y.get(2), 1e-15);
        assertEquals(0.0, y.get(16), 1e-15);
    }

    /**
     * Test of set method, of class SparseVector.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        
        x.set(0, 9.0);
        x.set(3, 9.0);
        x.set(7, 9.0);
        x.set(10, 9.0);
        
        x.set(1, 9.0);
        x.set(16, 9.0);
        
        assertEquals(9.0, x.get(0), 1e-15);
        assertEquals(9.0, x.get(3), 1e-15);
        assertEquals(9.0, x.get(7), 1e-15);
        assertEquals(9.0, x.get(10), 1e-15);
        
        assertEquals(3.0, x.get(15), 1e-15);
        assertEquals(2.0, x.get(18), 1e-15);
        assertEquals(5.0, x.get(19), 1e-15);

        assertEquals(9.0, x.get(1), 1e-15);
        assertEquals(0.0, x.get(2), 1e-15);
        assertEquals(9.0, x.get(16), 1e-15);
        
        
        //now the y values
        y.set(10, 9.0);
        y.set(15, 9.0);
        y.set(17, 9.0);
        y.set(18, 9.0);
        
        y.set(0, 9.0);
        y.set(16, 9.0);
        
        assertEquals(2.0, y.get(1), 1e-15);
        assertEquals(4.0, y.get(3), 1e-15);
        assertEquals(1.0, y.get(8), 1e-15);
        assertEquals(1.0, y.get(9), 1e-15);
        
        assertEquals(9.0, y.get(10), 1e-15);
        assertEquals(9.0, y.get(15), 1e-15);
        assertEquals(9.0, y.get(17), 1e-15);
        assertEquals(9.0, y.get(18), 1e-15);

        assertEquals(9.0, y.get(0), 1e-15);
        assertEquals(0.0, y.get(2), 1e-15);
        assertEquals(9.0, y.get(16), 1e-15);
    }

    /**
     * Test of sortedCopy method, of class SparseVector.
     */
    @Test
    public void testSortedCopy()
    {
        System.out.println("sortedCopy");
        Vec sorted = x.sortedCopy();
        double lastVal = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < sorted.length(); i++)
        {
            assertTrue(lastVal <= sorted.get(i));
            lastVal = sorted.get(i);
        }
    }

    /**
     * Test of min method, of class SparseVector.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        assertEquals(0.0, x.min(), 1e-15);
        
        y.set(15, -5.0);
        assertEquals(-5.0, y.min(), 1e-15);
    }

    /**
     * Test of max method, of class SparseVector.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        assertEquals(5.0, x.max(), 1e-15);
        
        y.set(15, -5.0);
        assertEquals(5.0, y.max(), 1e-15);
    }

    /**
     * Test of sum method, of class SparseVector.
     */
    @Test
    public void testSum()
    {
        System.out.println("sum");
        assertEquals(19.0, x.sum(), 1e-15);
        
        assertEquals(20.0, y.sum(), 1e-15);
    }

    /**
     * Test of variance method, of class SparseVector.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        assertEquals(2.2475, x.variance(), 1e-14);
        
        assertEquals(2.2, y.variance(), 1e-14);
    }

    /**
     * Test of median method, of class SparseVector.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        
        assertEquals(0.0, x.median(), 1e-15);
        assertEquals(0.0, y.median(), 1e-15);
        
        x.mutableAdd(y);
        assertEquals(1.0, x.median(), 1e-15);
    }

    /**
     * Test of skewness method, of class SparseVector.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        assertEquals(1.53870671976305, x.skewness(), 1e-14);
        
        assertEquals(1.49347441061993, y.skewness(), 1e-14);
    }

    /**
     * Test of kurtosis method, of class SparseVector.
     */
    @Test
    public void testKurtosis()
    {
        System.out.println("kurtosis");

        assertEquals(0.83542831548092615574590, x.kurtosis(), 1e-14);
        
        assertEquals(0.80165289256198347107438, y.kurtosis(), 1e-14);
    }

    /**
     * Test of dot method, of class SparseVector.
     */
    @Test
    public void testDot()
    {
        System.out.println("dot");
        assertEquals(6.30000000000000e+01, x.dot(x), 1e-14);
        
        assertEquals(6.40000000000000e+01, y.dot(y), 1e-14);
        
        assertEquals(3.90000000000000e+01, y.dot(x), 1e-14);
        assertEquals(3.90000000000000e+01, x.dot(y), 1e-14);
    }

    /**
     * Test of multiply method, of class SparseVector.
     */
    @Test
    public void testMultiply()
    {
        System.out.println("multiply");
        double c = 0.5;
        Matrix A = Matrix.pascal(20);
        Vec b = new DenseVector(20);
        
        x.multiply(c, A, b);
        
        DenseVector truth = new DenseVector(new double[]
        {
            9.50000000000000e+00,
            1.17000000000000e+02,
            1.02400000000000e+03,
            6.79100000000000e+03,
            3.65035000000000e+04,
            1.66677000000000e+05,
            6.67777000000000e+05,
            2.40242100000000e+06,
            7.89524000000000e+06,
            2.40138700000000e+07,
            6.82963280000000e+07,
            1.83124137000000e+08,
            4.66043078500000e+08,
            1.13203426200000e+09,
            2.63681749700000e+09,
            5.91310906500000e+09,
            1.28100303500000e+10,
            2.68885048480000e+10,
            5.48258186510000e+10,
            1.08840104671000e+11,
        });
        
        assertTrue(b.equals(truth, 1e-13));
    }

    /**
     * Test of mutableAdd method, of class SparseVector.
     */
    @Test
    public void testMutableAdd_double()
    {
        System.out.println("mutableAdd");
        
        x.mutableAdd(1.0);
        
        DenseVector truth = new DenseVector(new double[]{
            3.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            5.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            2.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            3.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            4.00000000000000e+00,
            1.00000000000000e+00,
            1.00000000000000e+00,
            3.00000000000000e+00,
            6.00000000000000e+00,
        });
        assertTrue(x.equals(truth, 1e-15));
    }

    /**
     * Test of mutableAdd method, of class SparseVector.
     */
    @Test
    public void testMutableAdd_double_Vec()
    {
        System.out.println("mutableAdd");
        double c = 0.5;
        
        x.mutableAdd(c, y);
        
        DenseVector truth = new DenseVector(new double[]{
            2.00000000000000e+00,
            1.00000000000000e+00,
            0.00000000000000e+00,
            6.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.00000000000000e+00,
            5.00000000000000e-01,
            5.00000000000000e-01,
            3.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            4.50000000000000e+00,
            0.00000000000000e+00,
            1.00000000000000e+00,
            4.50000000000000e+00,
            5.00000000000000e+00,
        });
        
        assertTrue(x.equals(truth, 1e-15));
    }

    /**
     * Test of mutableMultiply method, of class SparseVector.
     */
    @Test
    public void testMutableMultiply()
    {
        System.out.println("mutableMultiply");
        
        x.mutableMultiply(2);
        DenseVector truth = new DenseVector(new double[]{
            4.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            8.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            2.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            4.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            6.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            4.00000000000000e+00,
            1.00000000000000e+01, 
        });
        
        assertTrue(x.equals(truth, 1e-15));
    }

    /**
     * Test of mutableDivide method, of class SparseVector.
     */
    @Test
    public void testMutableDivide()
    {
        System.out.println("mutableDivide");

        x.mutableDivide(2);
        DenseVector truth = new DenseVector(new double[]{
            1.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            2.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            5.00000000000000e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.50000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.00000000000000e+00,
            2.50000000000000e+00,
        });
        
        assertTrue(x.equals(truth, 1e-15));
    }

    /**
     * Test of pNormDist method, of class SparseVector.
     */
    @Test
    public void testPNormDist()
    {
        System.out.println("pNormDist");
        assertEquals(1.70000000000000e+01, x.pNormDist(1, y), 1e-14);
        
        assertEquals(7.00000000000000e+00, x.pNormDist(2, y), 1e-14);
        
        assertEquals(5.24534393385174e+00, x.pNormDist(4, y), 1e-14);
        
        assertEquals(1.70000000000000e+01, y.pNormDist(1, x), 1e-14);
        
        assertEquals(7.00000000000000e+00, y.pNormDist(2, x), 1e-14);
        
        assertEquals(5.24534393385174e+00, y.pNormDist(4, x), 1e-14);
        
        
        assertEquals(0, x.pNormDist(4, x), 1e-14);
        
        assertEquals(0, y.pNormDist(1, y), 1e-14);
    }

    /**
     * Test of pNorm method, of class SparseVector.
     */
    @Test
    public void testPNorm()
    {
        System.out.println("pNorm");
        assertEquals(1.90000000000000e+01, x.pNorm(1), 1e-14);
        
        assertEquals(7.93725393319377e+00, x.pNorm(2), 1e-14);
        
        assertEquals(5.63881425400494e+00, x.pNorm(4), 1e-14);
        
        assertEquals(2.00000000000000e+01, y.pNorm(1), 1e-14);
        
        assertEquals(8.00000000000000e+00, y.pNorm(2), 1e-14);
        
        assertEquals(5.64020810264779e+00, y.pNorm(4), 1e-14);
    }

    /**
     * Test of clone method, of class SparseVector.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        
        SparseVector yClone = y.clone();
        yClone.mutableSubtract(y);
        yClone.mutableAdd(x);
        
        assertEquals(3.90000000000000e+01, y.dot(yClone), 1e-14);
    }

    /**
     * Test of normalize method, of class SparseVector.
     */
    @Test
    public void testNormalize()
    {
        System.out.println("normalize");
        
        x.normalize();
        
        DenseVector truth = new DenseVector(new double[]{
            2.51976315339485e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            5.03952630678970e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.25988157669742e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            2.51976315339485e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            3.77964473009227e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            2.51976315339485e-01,
            6.29940788348712e-01, 
        });
        
        assertTrue(x.equals(truth, 1e-14));
    }

    /**
     * Test of mutablePairwiseMultiply method, of class SparseVector.
     */
    @Test
    public void testMutablePairwiseMultiply()
    {
        System.out.println("mutablePairwiseMultiply");
        
        DenseVector truth = new DenseVector(new double[]{
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.60000000000000e+01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            4.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            9.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.00000000000000e+01,
            0.00000000000000e+00,
        });
        
        Vec result = x.pairwiseMultiply(y);
        
        assertTrue(truth.equals(result, 1e-14));
    }

    /**
     * Test of mutablePairwiseDivide method, of class SparseVector.
     */
    @Test
    public void testMutablePairwiseDivide()
    {
        System.out.println("mutablePairwiseDivide");
        DenseVector truth = new DenseVector(new double[]{
            2.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            8.00000000000000e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            6.66666666666667e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            7.50000000000000e-01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            3.33333333333333e-01,
            5.00000000000000e+00,
        });
        
        y.mutableAdd(1);//avoid dividion by zero
        x.mutablePairwiseDivide(y);
        
        assertTrue(truth.equals(x, 1e-14));
    }

    /**
     * Test of equals method, of class SparseVector.
     */
    @Test
    public void testEquals_Object()
    {
        System.out.println("equals");
        assertTrue(y.equals(y));
        assertTrue(x.equals(x));
        assertTrue(x.equals(new DenseVector(x)));
        
        assertFalse(y.equals(x));
        
        assertFalse(x.equals(y));
        
    }

    /**
     * Test of equals method, of class SparseVector.
     */
    @Test
    public void testEquals_Object_double()
    {
        System.out.println("equals");
        
        assertFalse(x.equals(y, 1e-15));
        
        assertFalse(x.equals(y, 4.5));
        
        assertTrue(x.equals(y, 5.5));
        
        assertTrue(x.equals(y, 10));
    }

    /**
     * Test of arrayCopy method, of class SparseVector.
     */
    @Test
    public void testArrayCopy()
    {
        System.out.println("arrayCopy");
        
        double[] xArray = x.arrayCopy();
        
        for(int i = 0; i < x.length(); i++)
            assertEquals(x.get(i), xArray[i], 0.0);
    }

    /**
     * Test of applyFunction method, of class SparseVector.
     */
    @Test
    public void testApplyFunction()
    {
        System.out.println("applyFunction");
        Function f = new FunctionBase() {

            /**
			 * 
			 */
			private static final long serialVersionUID = 5260043973153571531L;

			@Override
            public double f(Vec x)
            {
                return -x.get(0);
            }
        };
        
        x.applyFunction(f);
        
        DenseVector truth = new DenseVector(new double[]{
            -2.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -4.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -1.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -2.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -3.00000000000000e+00,
            -0.00000000000000e+00,
            -0.00000000000000e+00,
            -2.00000000000000e+00,
            -5.00000000000000e+00,
        });
        
        assertTrue(x.equals(truth, 1e-20));
    }

    /**
     * Test of applyIndexFunction method, of class SparseVector.
     */
    @Test
    public void testApplyIndexFunction()
    {
        System.out.println("applyIndexFunction");
        IndexFunction f = new IndexFunction() {

            /**
			 * 
			 */
			private static final long serialVersionUID = 2804170945957432993L;

			@Override
            public double indexFunc(double value, int index)
            {
                if(index < 0)
                    return 0;
                return value*index;
            }
        };
        
        x.applyIndexFunction(f);
        
        DenseVector truth = new DenseVector(new double[]{
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            1.20000000000000e+01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            7.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            2.00000000000000e+01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            0.00000000000000e+00,
            4.50000000000000e+01,
            0.00000000000000e+00,
            0.00000000000000e+00,
            3.60000000000000e+01,
            9.50000000000000e+01,
        });
        
        assertEquals(6, x.nnz());
    }

    /**
     * Test of zeroOut method, of class SparseVector.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        
        x.zeroOut();
        
        for(int i = 0; i < x.length(); i++)
            assertEquals(0.0, x.get(i), 0.0);
    }

    /**
     * Test of getNonZeroIterator method, of class SparseVector.
     */
    @Test
    public void testGetNonZeroIterator()
    {
        System.out.println("getNonZeroIterator");
        
        assertEquals(0, x.getNonZeroIterator(0).next().getIndex());
        assertEquals(1, y.getNonZeroIterator(0).next().getIndex());
        
        assertEquals(18, x.getNonZeroIterator(16).next().getIndex());
        assertEquals(17, y.getNonZeroIterator(16).next().getIndex());
        
        assertEquals(19, x.getNonZeroIterator(19).next().getIndex());
        assertFalse(y.getNonZeroIterator(19).hasNext());
    }

    /**
     * Test of hashCode method, of class SparseVector.
     */
    @Test
    public void testHashCode()
    {
        System.out.println("hashCode");
        assertEquals(new DenseVector(y).hashCode(), y.hashCode());
        assertEquals(new DenseVector(x).hashCode(), x.hashCode());
    }

    /**
     * Test of isSparse method, of class SparseVector.
     */
    @Test
    public void testIsSparse()
    {
        System.out.println("isSparse");
        assertTrue(x.isSparse());
        assertTrue(y.isSparse());
    }
}
