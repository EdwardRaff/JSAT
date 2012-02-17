/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear;

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
public class CholeskyDecompositionTest
{
    Matrix pascal4;
    Matrix pascal5;
    static ExecutorService threadpool = Executors.newFixedThreadPool(SystemInfo.LogicalCores+1, new ThreadFactory() {

        public Thread newThread(Runnable r)
        {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            return thread;
        }
    });
    
    public CholeskyDecompositionTest()
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
        pascal5 = new DenseMatrix(new double[][]
        {
            {    1,    1,    1,    1,    1},
            {    1,    2,    3,    4,    5},
            {    1,    3,    6,   10,   15},
            {    1,    4,   10,   20,   35},
            {    1,    5,   15,   35,   70}
        });
        
        pascal4 = new SubMatrix(pascal5, 0, 0, 4, 4).clone();
    }

    @Test
    public void testGetL()
    {
        Matrix p5L = new DenseMatrix(new double[][]
        {
            {1,     1,     1,     1,     1},
            {0,     1,     2,     3,     4},
            {0,     0,     1,     3,     6},
            {0,     0,     0,     1,     4},
            {0,     0,     0,     0,     1}
        });
        
        Matrix p4L = new DenseMatrix(new double[][]
        {
            {1,     1,     1,     1},
            {0,     1,     2,     3},
            {0,     0,     1,     3},
            {0,     0,     0,     1}
        });
        
        assertTrue(p4L.equals(new CholeskyDecomposition(pascal4).getLT(), 1e-10));
        assertTrue(p5L.equals(new CholeskyDecomposition(pascal5).getLT(), 1e-10));
    }
    
    @Test
    public void testGetDet()
    {
        double detP5 = 1.0;
        double detP4 = 1.0;
        
        assertEquals(detP4, new CholeskyDecomposition(pascal4).getDet(), 1e-10);
        assertEquals(detP5, new CholeskyDecomposition(pascal5).getDet(), 1e-10);
    }
    
    @Test
    public void testSolveVec()
    {
        Vec b5 = DenseVector.toDenseVec(5, 4, 1, 3, 2);
        Vec b4 = DenseVector.toDenseVec(4, 2, 3, 1);
        Vec x5 = DenseVector.toDenseVec(-18, 84, -113, 67, -15);
        Vec x4 = DenseVector.toDenseVec(15, -26, 21, -6);
        
        assertTrue(x5.equals(new CholeskyDecomposition(pascal5).solve(b5), 1e-10));
        assertTrue(x4.equals(new CholeskyDecomposition(pascal4).solve(b4), 1e-10));
    }
    
    @Test
    public void testSolveMatrix()
    {
        Matrix B5 = new DenseMatrix(new double[][]{
            {     2,     3,     5},
            {     1,     7,     1},
            {     4,     5,     5},
            {     3,     9,     1},
            {     2,     3,     8}
        });
        
        Matrix B4 = new DenseMatrix(new double[][]
        {
            {     4,     7,     6},
            {     3,     3,     5},
            {     5,     7,     3},
            {     7,     4,     6},
        });
        Matrix X5 = new DenseMatrix(new double[][]{
            {    27,   -47,    68},
            {   -81,   164,  -208},
            {   100,  -210,   266},
            {   -56,   124,  -156},
            {    12,   -28,    35}
        });
        
        Matrix X4 = new DenseMatrix(new double[][]
        {
            {	 11,   34,      0},
            {   -16,   -65,    19},
            {    12,    53,   -19},
            {    -3,   -15,     6},
        });
        
        assertTrue(X5.equals(new CholeskyDecomposition(pascal5).solve(B5), 1e-10));
        assertTrue(X4.equals(new CholeskyDecomposition(pascal4).solve(B4), 1e-10));
    }
    
    @Test
    public void testSolveMatrixExecutor()
    {
        Matrix B5 = new DenseMatrix(new double[][]{
            {     2,     3,     5},
            {     1,     7,     1},
            {     4,     5,     5},
            {     3,     9,     1},
            {     2,     3,     8}
        });
        
        Matrix B4 = new DenseMatrix(new double[][]
        {
            {     4,     7,     6},
            {     3,     3,     5},
            {     5,     7,     3},
            {     7,     4,     6},
        });
        Matrix X5 = new DenseMatrix(new double[][]{
            {    27,   -47,    68},
            {   -81,   164,  -208},
            {   100,  -210,   266},
            {   -56,   124,  -156},
            {    12,   -28,    35}
        });
        
        Matrix X4 = new DenseMatrix(new double[][]
        {
            {	 11,   34,      0},
            {   -16,   -65,    19},
            {    12,    53,   -19},
            {    -3,   -15,     6},
        });
        
        assertTrue(X5.equals(new CholeskyDecomposition(pascal5).solve(B5, threadpool), 1e-10));
        assertTrue(X4.equals(new CholeskyDecomposition(pascal4).solve(B4, threadpool), 1e-10));
    }
}
