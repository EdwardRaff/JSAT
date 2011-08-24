/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class DenseMatrixTest
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
    
    static DenseMatrix AB;
    static DenseMatrix BA;
    static DenseMatrix AC;
    
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
    
    public DenseMatrixTest()
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
        
        AB = new DenseMatrix(new double[][] 
        {
            {110,   150,   117,   157,   121},
            {82,   105,   102,   121,   101},
            {74,   103,   106,   121,    92},
            {44,   136,    65,   135,   130},
            {91,   178,   110,   171,   148}
        } );
        
        BA = new DenseMatrix(new double[][] 
        {
            {40,   182,    73,   187,   126},
            {35,   174,   100,   161,   131},
            {22,   109,    74,   115,    83},
            {45,   201,   127,   193,   156},
            {21,   100,    87,   123,   123}
        } );
        
        AC = new DenseMatrix(new double[][] 
        {
            {139,    73,   113,   167,   135,   100,   187},
            {116,    54,   106,   143,   136,    81,   153},
            {130,    42,    95,   142,   136,    64,   158},
            {106,    79,    62,   114,    33,   123,    86},
            {149,    90,    99,   173,   103,   139,   152}
        } );
        
        
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }

    /**
     * Test of mutableAdd method, of class DenseMatrix.
     */
    @Test
    public void testMutableAdd_Matrix()
    {
        DenseMatrix ApB = new DenseMatrix(new double[][] 
        {
            {6,     8,     6,    16,    17},
            {2,    13,    10,     9,    15},
            {1,     5,    14,    10,    10},
            {6,    17,     5,    16,     6},
            {9,    12,     6,    12,     7}
        } );
        
        Matrix aCopy = A.copy();
        Matrix bCopy = B.copy();
        
        aCopy.mutableAdd(B);
        bCopy.mutableAdd(A);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        try
        {
            C.copy().mutableAdd(A);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableAdd method, of class DenseMatrix.
     */
    @Test
    public void testMutableAdd_Matrix_ExecutorService()
    {
        DenseMatrix ApB = new DenseMatrix(new double[][] 
        {
            {6,     8,     6,    16,    17},
            {2,    13,    10,     9,    15},
            {1,     5,    14,    10,    10},
            {6,    17,     5,    16,     6},
            {9,    12,     6,    12,     7}
        } );
        
        Matrix aCopy = A.copy();
        Matrix bCopy = B.copy();
        
        aCopy.mutableAdd(B, threadpool);
        bCopy.mutableAdd(A, threadpool);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        try
        {
            C.copy().mutableAdd(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableAdd method, of class DenseMatrix.
     */
    @Test
    public void testMutableAdd_double()
    {
        DenseMatrix ApTwo = new DenseMatrix(new double[][] 
        {
            {1+2, 5+2, 4+2, 8+2, 9+2},
            {1+2, 5+2, 7+2, 3+2, 7+2},
            {0+2, 3+2, 8+2, 5+2, 6+2},
            {3+2, 8+2, 0+2, 7+2, 0+2},
            {1+2, 9+2, 2+2, 9+2, 6+2}
        } );
        
        Matrix aCopy = A.copy();
        
        aCopy.mutableAdd(2);
        
        assertEquals(ApTwo, aCopy);
    }

    /**
     * Test of mutableAdd method, of class DenseMatrix.
     */
    @Test
    public void testMutableAdd_double_ExecutorService()
    {
        DenseMatrix ApTwo = new DenseMatrix(new double[][] 
        {
            {1+2, 5+2, 4+2, 8+2, 9+2},
            {1+2, 5+2, 7+2, 3+2, 7+2},
            {0+2, 3+2, 8+2, 5+2, 6+2},
            {3+2, 8+2, 0+2, 7+2, 0+2},
            {1+2, 9+2, 2+2, 9+2, 6+2}
        } );
        
        Matrix aCopy = A.copy();
        
        aCopy.mutableAdd(2, threadpool);
        
        assertEquals(ApTwo, aCopy);
    }

    /**
     * Test of mutableSubtract method, of class DenseMatrix.
     */
    @Test
    public void testMutableSubtract_Matrix()
    {
        DenseMatrix AmB = new DenseMatrix(new double[][] 
        {
            {-4,     2,     2,     0,     1},
            { 0,    -3,     4,    -3,    -1},
            {-1,     1,     2,     0,     2},
            { 0,    -1,    -5,    -2,    -6},
            {-7,     6,    -2,     6,     5}
        } );
        
        DenseMatrix BmA = new DenseMatrix(new double[][] 
        {
            {-4*-1,     2*-1,     2*-1,     0*-1,     1*-1},
            { 0*-1,    -3*-1,     4*-1,    -3*-1,    -1*-1},
            {-1*-1,     1*-1,     2*-1,     0*-1,     2*-1},
            { 0*-1,    -1*-1,    -5*-1,    -2*-1,    -6*-1},
            {-7*-1,     6*-1,    -2*-1,     6*-1,     5*-1}
        } );
        
        Matrix aCopy = A.copy();
        Matrix bCopy = B.copy();
        
        aCopy.mutableSubtract(B);
        bCopy.mutableSubtract(A);
        
        assertEquals(AmB, aCopy);
        assertEquals(BmA, bCopy);
        
        try
        {
            C.copy().mutableSubtract(A);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableSubtract method, of class DenseMatrix.
     */
    @Test
    public void testMutableSubtract_Matrix_ExecutorService()
    {
        DenseMatrix AmB = new DenseMatrix(new double[][] 
        {
            {-4,     2,     2,     0,     1},
            { 0,    -3,     4,    -3,    -1},
            {-1,     1,     2,     0,     2},
            { 0,    -1,    -5,    -2,    -6},
            {-7,     6,    -2,     6,     5}
        } );
        
        DenseMatrix BmA = new DenseMatrix(new double[][] 
        {
            {-4*-1,     2*-1,     2*-1,     0*-1,     1*-1},
            { 0*-1,    -3*-1,     4*-1,    -3*-1,    -1*-1},
            {-1*-1,     1*-1,     2*-1,     0*-1,     2*-1},
            { 0*-1,    -1*-1,    -5*-1,    -2*-1,    -6*-1},
            {-7*-1,     6*-1,    -2*-1,     6*-1,     5*-1}
        } );
        
        Matrix aCopy = A.copy();
        Matrix bCopy = B.copy();
        
        aCopy.mutableSubtract(B, threadpool);
        bCopy.mutableSubtract(A, threadpool);
        
        assertEquals(AmB, aCopy);
        assertEquals(BmA, bCopy);
        
        try
        {
            C.copy().mutableSubtract(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of multiply method, of class DenseMatrix.
     */
    @Test
    public void testMultiply_Vec()
    {
        fail("Not yet implemented");
    }

    /**
     * Test of multiply method, of class DenseMatrix.
     */
    @Test
    public void testMultiply_Vec_ExecutorService()
    {
        fail("Not yet implemented");
    }
    
    /**
     * Test of multiply method, of class DenseMatrix.
     */
    @Test
    public void testMultiply_Matrix()
    {
        Matrix result;
        
        result = A.multiply(B);
        assertEquals(AB, result);
        
        result = B.multiply(A);
        assertEquals(BA, result);
        
        result = A.multiply(C);
        assertEquals(AC, result);
        
        try
        {
            C.multiply(A);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of multiply method, of class DenseMatrix.
     */
    @Test
    public void testMultiply_Matrix_ExecutorService()
    {
        Matrix result;
        
        result = A.multiply(B, threadpool);
        assertEquals(AB, result);
        
        result = B.multiply(A, threadpool);
        assertEquals(BA, result);
        
        result = A.multiply(C, threadpool);
        assertEquals(AC, result);
        
        try
        {
            C.multiply(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableMultiply method, of class DenseMatrix.
     */
    @Test
    public void testMutableMultiply_double()
    {
        DenseMatrix AtTwo = new DenseMatrix(new double[][] 
        {
            {1*2, 5*2, 4*2, 8*2, 9*2},
            {1*2, 5*2, 7*2, 3*2, 7*2},
            {0*2, 3*2, 8*2, 5*2, 6*2},
            {3*2, 8*2, 0*2, 7*2, 0*2},
            {1*2, 9*2, 2*2, 9*2, 6*2}
        } );
        
        Matrix aCopy = A.copy();
        
        aCopy.mutableMultiply(2);
        
        assertEquals(AtTwo, aCopy);
    }

    /**
     * Test of mutableMultiply method, of class DenseMatrix.
     */
    @Test
    public void testMutableMultiply_double_ExecutorService()
    {
        DenseMatrix AtTwo = new DenseMatrix(new double[][] 
        {
            {1*2, 5*2, 4*2, 8*2, 9*2},
            {1*2, 5*2, 7*2, 3*2, 7*2},
            {0*2, 3*2, 8*2, 5*2, 6*2},
            {3*2, 8*2, 0*2, 7*2, 0*2},
            {1*2, 9*2, 2*2, 9*2, 6*2}
        } );
        
        Matrix aCopy = A.copy();
        
        aCopy.mutableMultiply(2, threadpool);
        
        assertEquals(AtTwo, aCopy);
    }

    /**
     * Test of transpose method, of class DenseMatrix.
     */
    @Test
    public void testTranspose()
    {
        DenseMatrix CTranspose = new DenseMatrix(new double[][] 
        {
            {1, 5, 8, 9, 1},
            {6, 5, 0, 3, 2},
            {8, 3, 5, 2, 6},
            {3, 7, 7, 7, 5},
            {1, 2, 9, 2, 8}, 
            {5, 10, 1, 4, 1},
            {10, 0, 8, 8, 9}
        } );
        
        assertEquals(CTranspose, C.transpose());
    }

    /**
     * Test of get method, of class DenseMatrix.
     */
    @Test
    public void testGet()
    {
        //Tests both
        testSet();
    }

    /**
     * Test of set method, of class DenseMatrix.
     */
    @Test
    public void testSet()
    {
        DenseMatrix toSet = new DenseMatrix(A.rows(), A.cols());
        
        for(int i = 0; i < A.rows(); i++)
            for(int j = 0; j < A.cols(); j++)
                toSet.set(i, j, A.get(i, j));
        
        assertEquals(A, toSet);
    }

    /**
     * Test of rows method, of class DenseMatrix.
     */
    @Test
    public void testRows()
    {
        assertEquals(5, A.rows());
    }

    /**
     * Test of cols method, of class DenseMatrix.
     */
    @Test
    public void testCols()
    {
        assertEquals(5, A.cols());
        assertEquals(7, C.cols());
    }

    /**
     * Test of isSparce method, of class DenseMatrix.
     */
    @Test
    public void testIsSparce()
    {
        assertEquals(false, A.isSparce());
    }

    /**
     * Test of nnz method, of class DenseMatrix.
     */
    @Test
    public void testNnz()
    {
        assertEquals(5*5, A.nnz());
        assertEquals(5*7, C.nnz());
    }

    /**
     * Test of copy method, of class DenseMatrix.
     */
    @Test
    public void testCopy()
    {
        Matrix ACopy = A.copy();
        
        assertEquals(A, ACopy);
        assertEquals(A.multiply(B), ACopy.multiply(B));
    }

    /**
     * Test of swapRows method, of class DenseMatrix.
     */
    @Test
    public void testSwapRows()
    {
        System.out.println("swapRows");
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of zeroOut method, of class DenseMatrix.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of lup method, of class DenseMatrix.
     */
    @Test
    public void testLup_0args()
    {
        System.out.println("lup");
        
        DenseMatrix A_L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.3333,    1.0000,         0,         0,         0},
            {     0,    0.4737,    1.0000,         0,         0},
            {0.3333,    0.3684,    0.8881,    1.0000,         0},
            {0.3333,    0.3684,    0.4627,   -0.6885,    1.0000}
        } );
        
        DenseMatrix A_U = new DenseMatrix(new double[][] 
        {
            {3.0000,    8.0000,         0,    7.0000,         0},
            {     0,    6.3333,    2.0000,    6.6667,    6.0000},
            {     0,         0,    7.0526,    1.8421,    3.1579},
            {     0,         0,         0,   -3.4254,    1.9851},
            {     0,         0,         0,         0,    6.6950}
        } );
        
        DenseMatrix A_P = new DenseMatrix(new double[][] 
        {
            {0,     0,     0,     1,     0},
            {0,     0,     0,     0,     1},
            {0,     0,     1,     0,     0},
            {0,     1,     0,     0,     0},
            {1,     0,     0,     0,     0}
        } );
        
        DenseMatrix C_L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.1111,    1.0000,         0,         0,         0},
            {0.8889,   -0.4706,    1.0000,         0,         0},
            {0.1111,    0.2941,    0.5071,    1.0000,         0},
            {0.5556,    0.5882,   -0.3903,    0.9515,    1.0000}
        } );
        
        DenseMatrix C_U = new DenseMatrix(new double[][] 
        {
            {9.0000,    3.0000,    2.0000,    7.0000,    2.0000,    4.0000,    8.0000},
            {     0,    5.6667,    7.7778,    2.2222,    0.7778,    4.5556,    9.1111},
            {     0,         0,    6.8824,    1.8235,    7.5882,   -0.4118,    5.1765},
            {     0,         0,         0,    2.6439,    3.7009,   -0.5755,    2.8063},
            {     0,         0,         0,         0,   -0.1282,    5.4849,  -10.4537}
        } );
        
        DenseMatrix C_P = new DenseMatrix(new double[][] 
        {
            {0,     0,     0,     1,     0},
            {1,     0,     0,     0,     0},
            {0,     0,     1,     0,     0},
            {0,     0,     0,     0,     1},
            {0,     1,     0,     0,     0}
        } );
        
        DenseMatrix CT_L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.5000,    1.0000,         0,         0,         0},
            {0.1000,    0.2000,    1.0000,         0,         0},
            {0.1000,    0.5000,    0.9886,    1.0000,         0},
            {0.8000,    0.3000,   -0.0568,   -0.6176,    1.0000},
            {0.6000,    0.5000,   -0.3750,   -0.1925,   -0.0441},
            {0.3000,    0.7000,    0.7614,    0.5256,   -0.5687}
        } );
        
        DenseMatrix CT_U = new DenseMatrix(new double[][] 
        {
            {10.0000,         0,    8.0000,    8.0000,    9.0000},
            {      0,   10.0000,   -3.0000,         0,   -3.5000},
            {      0,         0,    8.8000,    1.2000,    7.8000},
            {      0,         0,         0,    7.0136,   -5.8614},
            {      0,         0,         0,         0,   -3.3270}
        } );
        
        DenseMatrix CT_P = new DenseMatrix(new double[][] 
        {
            {0,     0,     0,     0,     0,     0,     1},
            {0,     0,     0,     0,     0,     1,     0},
            {0,     0,     0,     0,     1,     0,     0},
            {1,     0,     0,     0,     0,     0,     0},
            {0,     0,     1,     0,     0,     0,     0}, 
            {0,     1,     0,     0,     0,     0,     0},
            {0,     0,     0,     1,     0,     0,     0}
        } );
        
        Matrix[] lup;
        
        lup = A.copy().lup();
        assertTrue(lup[0].equalsRange(A_L, 0.0001));
        assertTrue(lup[1].equalsRange(A_U, 0.0001));
        assertTrue(lup[2].equalsRange(A_P, 0.0001));
        assertTrue(lup[2].multiply(A).equalsRange(lup[0].multiply(lup[1]), 1e-14));
        
        lup = C.copy().lup();
        assertTrue(lup[0].equalsRange(C_L, 0.0001));
        assertTrue(lup[1].equalsRange(C_U, 0.0001));
        assertTrue(lup[2].equalsRange(C_P, 0.0001));
        assertTrue(lup[2].multiply(C).equalsRange(lup[0].multiply(lup[1]), 1e-14));
        
        
        lup = C.transpose().lup();
        assertTrue(lup[0].equalsRange(CT_L, 0.0001));
        assertTrue(lup[1].equalsRange(CT_U, 0.0001));
        assertTrue(lup[2].equalsRange(CT_P, 0.0001));
        assertTrue(lup[2].multiply(C.transpose()).equalsRange(lup[0].multiply(lup[1]), 1e-14));
    }

    /**
     * Test of lup method, of class DenseMatrix.
     */
    @Test
    public void testLup_ExecutorService()
    {
        System.out.println("lup");
        DenseMatrix A_L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.3333,    1.0000,         0,         0,         0},
            {     0,    0.4737,    1.0000,         0,         0},
            {0.3333,    0.3684,    0.8881,    1.0000,         0},
            {0.3333,    0.3684,    0.4627,   -0.6885,    1.0000}
        } );
        
        DenseMatrix A_U = new DenseMatrix(new double[][] 
        {
            {3.0000,    8.0000,         0,    7.0000,         0},
            {     0,    6.3333,    2.0000,    6.6667,    6.0000},
            {     0,         0,    7.0526,    1.8421,    3.1579},
            {     0,         0,         0,   -3.4254,    1.9851},
            {     0,         0,         0,         0,    6.6950}
        } );
        
        DenseMatrix A_P = new DenseMatrix(new double[][] 
        {
            {0,     0,     0,     1,     0},
            {0,     0,     0,     0,     1},
            {0,     0,     1,     0,     0},
            {0,     1,     0,     0,     0},
            {1,     0,     0,     0,     0}
        } );
        
        DenseMatrix C_L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.1111,    1.0000,         0,         0,         0},
            {0.8889,   -0.4706,    1.0000,         0,         0},
            {0.1111,    0.2941,    0.5071,    1.0000,         0},
            {0.5556,    0.5882,   -0.3903,    0.9515,    1.0000}
        } );
        
        DenseMatrix C_U = new DenseMatrix(new double[][] 
        {
            {9.0000,    3.0000,    2.0000,    7.0000,    2.0000,    4.0000,    8.0000},
            {     0,    5.6667,    7.7778,    2.2222,    0.7778,    4.5556,    9.1111},
            {     0,         0,    6.8824,    1.8235,    7.5882,   -0.4118,    5.1765},
            {     0,         0,         0,    2.6439,    3.7009,   -0.5755,    2.8063},
            {     0,         0,         0,         0,   -0.1282,    5.4849,  -10.4537}
        } );
        
        DenseMatrix C_P = new DenseMatrix(new double[][] 
        {
            {0,     0,     0,     1,     0},
            {1,     0,     0,     0,     0},
            {0,     0,     1,     0,     0},
            {0,     0,     0,     0,     1},
            {0,     1,     0,     0,     0}
        } );
        
        DenseMatrix CT_L = new DenseMatrix(new double[][] 
        {
            {1.0000,         0,         0,         0,         0},
            {0.5000,    1.0000,         0,         0,         0},
            {0.1000,    0.2000,    1.0000,         0,         0},
            {0.1000,    0.5000,    0.9886,    1.0000,         0},
            {0.8000,    0.3000,   -0.0568,   -0.6176,    1.0000},
            {0.6000,    0.5000,   -0.3750,   -0.1925,   -0.0441},
            {0.3000,    0.7000,    0.7614,    0.5256,   -0.5687}
        } );
        
        DenseMatrix CT_U = new DenseMatrix(new double[][] 
        {
            {10.0000,         0,    8.0000,    8.0000,    9.0000},
            {      0,   10.0000,   -3.0000,         0,   -3.5000},
            {      0,         0,    8.8000,    1.2000,    7.8000},
            {      0,         0,         0,    7.0136,   -5.8614},
            {      0,         0,         0,         0,   -3.3270}
        } );
        
        DenseMatrix CT_P = new DenseMatrix(new double[][] 
        {
            {0,     0,     0,     0,     0,     0,     1},
            {0,     0,     0,     0,     0,     1,     0},
            {0,     0,     0,     0,     1,     0,     0},
            {1,     0,     0,     0,     0,     0,     0},
            {0,     0,     1,     0,     0,     0,     0}, 
            {0,     1,     0,     0,     0,     0,     0},
            {0,     0,     0,     1,     0,     0,     0}
        } );
        
        Matrix[] lup;
        
        lup = A.copy().lup(threadpool);
        assertTrue(lup[0].equalsRange(A_L, 0.0001));
        assertTrue(lup[1].equalsRange(A_U, 0.0001));
        assertTrue(lup[2].equalsRange(A_P, 0.0001));
        assertTrue(lup[2].multiply(A, threadpool).equalsRange(lup[0].multiply(lup[1], threadpool), 1e-14));
        
        lup = C.copy().lup(threadpool);
        assertTrue(lup[0].equalsRange(C_L, 0.0001));
        assertTrue(lup[1].equalsRange(C_U, 0.0001));
        assertTrue(lup[2].equalsRange(C_P, 0.0001));
        assertTrue(lup[2].multiply(C, threadpool).equalsRange(lup[0].multiply(lup[1], threadpool), 1e-14));
        
        
        lup = C.transpose().lup(threadpool);
        assertTrue(lup[0].equalsRange(CT_L, 0.0001));
        assertTrue(lup[1].equalsRange(CT_U, 0.0001));
        assertTrue(lup[2].equalsRange(CT_P, 0.0001));
        assertTrue(lup[2].multiply(C.transpose(), threadpool).equalsRange(lup[0].multiply(lup[1], threadpool), 1e-14));
    }
}
