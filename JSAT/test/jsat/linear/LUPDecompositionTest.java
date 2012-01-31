package jsat.linear;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.Executors;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import jsat.utils.SystemInfo;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class LUPDecompositionTest
{
    /**
     * 5x5
     */
    static DenseMatrix A;
    /**
     * 5x5
     */
    static DenseMatrix B;
    /**
     * 5x7
     */
    static DenseMatrix C;
    
    static ExecutorService threadpool = Executors.newFixedThreadPool(SystemInfo.LogicalCores+1, new ThreadFactory() {

        public Thread newThread(Runnable r)
        {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            return thread;
        }
    });
    
    public LUPDecompositionTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        A = new DenseMatrix(new double[][] 
        {
            {1, 5, 4, 8, 9},
            {1, 5, 7, 3, 7},
            {0, 3, 8, 5, 6},
            {3, 8, 0, 7, 0},
            {1, 9, 2, 9, 6}
        } );
        
        B = new DenseMatrix(new double[][] 
        {
            {5, 3, 2, 8, 8},
            {1, 8, 3, 6, 8},
            {1, 2, 6, 5, 4},
            {3, 9, 5, 9, 6},
            {8, 3, 4, 3, 1}
        } );
        
        C = new DenseMatrix(new double[][] 
        {
            {1, 6, 8, 3, 1, 5, 10},
            {5, 5, 3, 7, 2, 10, 0},
            {8, 0, 5, 7, 9, 1, 8},
            {9, 3, 2, 7, 2, 4, 8},
            {1, 2, 6, 5, 8, 1, 9}
        } );
        
        
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
     * Test of isSquare method, of class LUPDecomposition.
     */
    @Test
    public void testIsSquare()
    {
        System.out.println("isSquare");
        LUPDecomposition instance = new LUPDecomposition(C);
        
        assertFalse(instance.isSquare());
        
        instance = new LUPDecomposition(C.transpose());
        
        assertFalse(instance.isSquare());
        
        instance = new LUPDecomposition(A);
        
        assertTrue(instance.isSquare());
    }

    /**
     * Test of det method, of class LUPDecomposition.
     */
    @Test
    public void testDet()
    {
        System.out.println("det");
        LUPDecomposition instance = null;
        
        instance = new LUPDecomposition(A);
        assertEquals(3073, instance.det(), 1e-10);
        
        instance = new LUPDecomposition(B);
        assertEquals(-8068, instance.det(), 1e-10);
        
        instance = new LUPDecomposition(C);
        try
        {
            instance.det();
            fail("Can not take the determinant of a non square matrix");
        }
        catch(ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of solve method, of class LUPDecomposition.
     */
    @Test
    public void testSolve_Vec()
    {
        System.out.println("solve");
        Vec b = new DenseVector(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0 ));
        LUPDecomposition instance = null;
        
        instance = new LUPDecomposition(A);
        Vec x = instance.solve(b);
        assertTrue(A.multiply(x).equals(b, 1e-10));
        
        instance = new LUPDecomposition(C);
        x = instance.solve(b);
        assertTrue(C.multiply(x).equals(b, 1e-10));
        
    }

    /**
     * Test of solve method, of class LUPDecomposition.
     */
    @Test
    public void testSolve_Matrix()
    {
        System.out.println("solve");
        LUPDecomposition instance = null;
        Matrix x;
        
        instance = new LUPDecomposition(A);
        x = instance.solve(B);
        assertTrue(A.multiply(x).equals(B, 1e-10));
        
        instance = new LUPDecomposition(A);
        x = instance.solve(C);
        assertTrue(A.multiply(x).equals(C, 1e-10));
        
        instance = new LUPDecomposition(C);
        x = instance.solve(A);
        assertTrue(C.multiply(x).equals(A, 1e-10));
        
        instance = new LUPDecomposition(C.transpose());
        try
        {
            instance.solve(A);
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
        
        instance = new LUPDecomposition(A);
        try
        {
            instance.solve(C.transpose());
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of solve method, of class LUPDecomposition.
     */
    @Test
    public void testSolve_Matrix_ExecutorService()
    {
        System.out.println("solve");
        LUPDecomposition instance = null;
        Matrix x;
        
        instance = new LUPDecomposition(A);
        x = instance.solve(B, threadpool);
        assertTrue(A.multiply(x).equals(B, 1e-10));
        
        instance = new LUPDecomposition(A);
        x = instance.solve(C, threadpool);
        assertTrue(A.multiply(x).equals(C, 1e-10));
        
        instance = new LUPDecomposition(C);
        x = instance.solve(A, threadpool);
        assertTrue(C.multiply(x).equals(A, 1e-10));
        
        instance = new LUPDecomposition(C.transpose());
        try
        {
            instance.solve(A, threadpool);
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
        
        instance = new LUPDecomposition(A);
        try
        {
            instance.solve(C.transpose(), threadpool);
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of forwardSub method, of class LUPDecomposition.
     */
    @Test
    public void testForwardSub_Matrix_Vec()
    {
        System.out.println("forwardSub");
        DenseMatrix L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.1111,    1.0000,         0,         0,         0},
            {0.1111,    0.2941,    1.0000,         0,         0},
            {0.5556,    0.5882,   -0.7697,    1.0000,         0},
            {0.8889,   -0.4706,    1.9719,   -1.1457,    1.0000}
        } );
        
        Vec b = new DenseVector(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0 ));
        
        
        Vec x = LUPDecomposition.forwardSub(L, b);
        
        assertTrue(L.multiply(x).equals(b, 1e-10));
        
       
       
       try
       {
           b = new DenseVector(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0 , 6.0));
           LUPDecomposition.forwardSub(L, b);
           fail("Dimensions dont agree!");
       }
       catch(ArithmeticException ex)
       {
           
       }
       
       try
       {
           b = new DenseVector(Arrays.asList(1.0, 2.0, 3.0, 4.0));
           LUPDecomposition.forwardSub(L, b);
           fail("Dimensions dont agree!");
       }
       catch(ArithmeticException ex)
       {
           
       }
        
    }

    /**
     * Test of forwardSub method, of class LUPDecomposition.
     */
    @Test
    public void testForwardSub_Matrix_Matrix()
    {
        System.out.println("forwardSub");
        
        DenseMatrix L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.1111,    1.0000,         0,         0,         0},
            {0.1111,    0.2941,    1.0000,         0,         0},
            {0.5556,    0.5882,   -0.7697,    1.0000,         0},
            {0.8889,   -0.4706,    1.9719,   -1.1457,    1.0000}
        } );
        
        DenseMatrix b = new DenseMatrix(new double[][] 
        {
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10}
        } );
        
        Matrix x = LUPDecomposition.forwardSub(L, b);
        assertTrue(L.multiply(x).equals(b, 1e-10));
    }

    /**
     * Test of forwardSub method, of class LUPDecomposition.
     */
    @Test
    public void testForwardSub_3args()
    {
        System.out.println("forwardSub");
        DenseMatrix L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.1111,    1.0000,         0,         0,         0},
            {0.1111,    0.2941,    1.0000,         0,         0},
            {0.5556,    0.5882,   -0.7697,    1.0000,         0},
            {0.8889,   -0.4706,    1.9719,   -1.1457,    1.0000}
        } );
        
        DenseMatrix b = new DenseMatrix(new double[][] 
        {
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10}
        } );
        
        Matrix x = LUPDecomposition.forwardSub(L, b, threadpool);
        assertTrue(L.multiply(x).equals(b, 1e-10));
    }

    /**
     * Test of backSub method, of class LUPDecomposition.
     */
    @Test
    public void testBackSub_Matrix_Vec()
    {
        System.out.println("backSub");
        DenseMatrix U = new DenseMatrix(new double[][] 
        {
            {9.0000,    3.0000,    2.0000,    7.0000,    2.0000},
            {     0,    5.6667,    7.7778,    2.2222,    0.7778},
            {     0,         0,    3.4902,    3.5686,    7.5490},
            {     0,         0,         0,    4.5506,    6.2416},
            {     0,         0,         0,         0,   -0.1469}
        } );
        
        Vec b = new DenseVector(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0 ));
        
        
        Vec x = LUPDecomposition.backSub(U, b);
        
        assertTrue(U.multiply(x).equals(b, 1e-10));
    }

    /**
     * Test of backSub method, of class LUPDecomposition.
     */
    @Test
    public void testBackSub_Matrix_Matrix()
    {
        System.out.println("backSub");
        DenseMatrix U = new DenseMatrix(new double[][] 
        {
            {9.0000,    3.0000,    2.0000,    7.0000,    2.0000},
            {     0,    5.6667,    7.7778,    2.2222,    0.7778},
            {     0,         0,    3.4902,    3.5686,    7.5490},
            {     0,         0,         0,    4.5506,    6.2416},
            {     0,         0,         0,         0,   -0.1469}
        } );
        
        DenseMatrix b = new DenseMatrix(new double[][] 
        {
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10}
        } );
        
        Matrix x = LUPDecomposition.backSub(U, b);
        assertTrue(U.multiply(x).equals(b, 1e-10));
    }

    /**
     * Test of backSub method, of class LUPDecomposition.
     */
    @Test
    public void testBackSub_3args()
    {
        System.out.println("backSub");
        DenseMatrix U = new DenseMatrix(new double[][] 
        {
            {9.0000,    3.0000,    2.0000,    7.0000,    2.0000},
            {     0,    5.6667,    7.7778,    2.2222,    0.7778},
            {     0,         0,    3.4902,    3.5686,    7.5490},
            {     0,         0,         0,    4.5506,    6.2416},
            {     0,         0,         0,         0,   -0.1469}
        } );
        
        DenseMatrix b = new DenseMatrix(new double[][] 
        {
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10}
        } );
        
        Matrix x = LUPDecomposition.backSub(U, b, threadpool);
        assertTrue(U.multiply(x).equals(b, 1e-10));
    }
}
