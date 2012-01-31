package jsat.linear;

import java.util.Arrays;
import jsat.utils.SystemInfo;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class QRDecompositionTest
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
    
    public QRDecompositionTest()
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
     * Test of absDet method, of class QRDecomposition.
     */
    @Test
    public void testAbsDet()
    {
        System.out.println("absDet");
        QRDecomposition instance = null;
        
        instance = new QRDecomposition(A);
        assertEquals(3073, instance.absDet(), 1e-10);
        
        instance = new QRDecomposition(B);
        assertEquals(8068, instance.absDet(), 1e-10);
        
        instance = new QRDecomposition(C);
        try
        {
            instance.absDet();
            fail("Can not take the determinant of a non square matrix");
        }
        catch(ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of solve method, of class QRDecomposition.
     */
    @Test
    public void testSolve_Vec()
    {
        System.out.println("solve");
        Vec b = new DenseVector(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0 ));
        QRDecomposition instance = null;
        
        instance = new QRDecomposition(A);
        Vec x = instance.solve(b);
        assertTrue(A.multiply(x).equals(b, 1e-10));
        
        instance = new QRDecomposition(C);
        x = instance.solve(b);
        assertTrue(C.multiply(x).equals(b, 1e-10));
    }

    /**
     * Test of solve method, of class QRDecomposition.
     */
    @Test
    public void testSolve_Matrix()
    {
        System.out.println("solve");
        QRDecomposition instance = null;
        Matrix x;
        
        instance = new QRDecomposition(A);
        x = instance.solve(B);
        assertTrue(A.multiply(x).equals(B, 1e-10));
        
        instance = new QRDecomposition(A);
        x = instance.solve(C);
        assertTrue(A.multiply(x).equals(C, 1e-10));
        
        instance = new QRDecomposition(C);
        x = instance.solve(A);
        assertTrue(C.multiply(x).equals(A, 1e-10));
        
        instance = new QRDecomposition(C.transpose());
        try
        {
            instance.solve(A);
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
        
        instance = new QRDecomposition(A);
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
     * Test of solve method, of class QRDecomposition.
     */
    @Test
    public void testSolve_Matrix_ExecutorService()
    {
        System.out.println("solve");
        QRDecomposition instance = null;
        Matrix x;
        
        instance = new QRDecomposition(A);
        x = instance.solve(B, threadpool);
        assertTrue(A.multiply(x).equals(B, 1e-10));
        
        instance = new QRDecomposition(A);
        x = instance.solve(C, threadpool);
        assertTrue(A.multiply(x).equals(C, 1e-10));
        
        instance = new QRDecomposition(C);
        x = instance.solve(A, threadpool);
        assertTrue(C.multiply(x).equals(A, 1e-10));
        
        instance = new QRDecomposition(C.transpose());
        try
        {
            instance.solve(A, threadpool);
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
        
        instance = new QRDecomposition(A);
        try
        {
            instance.solve(C.transpose(), threadpool);
            fail("The matrix dimensions do not agree!");
        }
        catch(ArithmeticException ex)
        {
            
        }
    }
}
