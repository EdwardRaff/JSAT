package jsat.linear;

import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
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
public class HessenbergFormTest
{
    
    /**
     * 5x5
     */
    static DenseMatrix A, hessA;
    
    /**
     * 2x2 test matrix
     */
    static DenseMatrix f;
    
    
    /**
     * Multi threaded pool with daemon threads
     */
    static ExecutorService threadpool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()+1, new ThreadFactory() {

        public Thread newThread(Runnable r)
        {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            return thread;
        }
    });
    
    
    public HessenbergFormTest()
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
        
        hessA = new DenseMatrix(new double[][] 
        {
            { 1.000000000000000,  -11.457431093955016,  -7.374707356495900,  -0.099215299078182,   0.575430671538097},
            { -3.316624790355400,  13.636363636363633,   3.580749763336201,  -7.121112733469424,  -1.875518953902302},
            {              0,      11.971729785189224,  10.034550906364995,  -2.235016742983057,   1.182382333254766},
            {              0,                       0,  -7.708011539446948,  -0.430619433926174,  -3.219873923950582},
            {              0,                       0,                   0,   0.098549849397624,   2.759704891197538}
        });
        
        f = new DenseMatrix(new double[][] 
        {
            {8, 5},
            {6, 9}
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
     * Test of hess method, of class HessenbergForm.
     */
    @Test
    public void testHess_Matrix()
    {
        System.out.println("hess");
        
        Matrix test; 
        
        test = A.clone();
        HessenbergForm.hess(test);
        assertTrue(test.equals(hessA, 1e-14));
        
        //f is 2x2, so it is its own upper hessenberg forn
        test = f.clone();
        HessenbergForm.hess(f);
        assertEquals(test, f);
        
    }
    
    @Test
    public void testHess_Matrix_Executor()
    {
        System.out.println("hess");
        
        Matrix test; 
        
        test = A.clone();
        HessenbergForm.hess(test, threadpool);
        assertTrue(test.equals(hessA, 1e-14));
        
        //f is 2x2, so it is its own upper hessenberg forn
        test = f.clone();
        HessenbergForm.hess(f, threadpool);
        assertEquals(test, f);
        
    }
}
