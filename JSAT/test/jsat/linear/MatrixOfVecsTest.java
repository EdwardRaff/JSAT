/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package jsat.linear;

import java.util.Arrays;
import java.util.concurrent.*;
import org.junit.*;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class MatrixOfVecsTest
{
    /**
     * 5x5
     */
    static MatrixOfVecs A;
    /**
     * 5x5
     */
    static MatrixOfVecs B;
    /**
     * 5x7
     */
    static MatrixOfVecs C;
    
    static MatrixOfVecs AB;
    static MatrixOfVecs BA;
    static MatrixOfVecs AC;
    
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
    
    public MatrixOfVecsTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        A = new MatrixOfVecs(new Vec[] 
        {
            DenseVector.toDenseVec(1, 5, 4, 8, 9),
            DenseVector.toDenseVec(1, 5, 7, 3, 7),
            DenseVector.toDenseVec(0, 3, 8, 5, 6),
            DenseVector.toDenseVec(3, 8, 0, 7, 0),
            DenseVector.toDenseVec(1, 9, 2, 9, 6)
        } );
        
        B = new MatrixOfVecs(new Vec[] 
        {
            DenseVector.toDenseVec(5, 3, 2, 8, 8),
            DenseVector.toDenseVec(1, 8, 3, 6, 8),
            DenseVector.toDenseVec(1, 2, 6, 5, 4),
            DenseVector.toDenseVec(3, 9, 5, 9, 6),
            DenseVector.toDenseVec(8, 3, 4, 3, 1)
        } );
        
        C = new MatrixOfVecs(new Vec[] 
        {
            DenseVector.toDenseVec(1, 6, 8, 3, 1, 5, 10),
            DenseVector.toDenseVec(5, 5, 3, 7, 2, 10, 0),
            DenseVector.toDenseVec(8, 0, 5, 7, 9, 1, 8),
            DenseVector.toDenseVec(9, 3, 2, 7, 2, 4, 8),
            DenseVector.toDenseVec(1, 2, 6, 5, 8, 1, 9)
        } );
        
        AB = new MatrixOfVecs(new Vec[] 
        {
            DenseVector.toDenseVec(110,   150,   117,   157,   121),
            DenseVector.toDenseVec(82,   105,   102,   121,   101),
            DenseVector.toDenseVec(74,   103,   106,   121,    92),
            DenseVector.toDenseVec(44,   136,    65,   135,   130),
            DenseVector.toDenseVec(91,   178,   110,   171,   148)
        } );
        
        BA = new MatrixOfVecs(new Vec[] 
        {
            DenseVector.toDenseVec(40,   182,    73,   187,   126),
            DenseVector.toDenseVec(35,   174,   100,   161,   131),
            DenseVector.toDenseVec(22,   109,    74,   115,    83),
            DenseVector.toDenseVec(45,   201,   127,   193,   156),
            DenseVector.toDenseVec(21,   100,    87,   123,   123)
        } );
        
        AC = new MatrixOfVecs(new Vec[] 
        {
            DenseVector.toDenseVec(139,    73,   113,   167,   135,   100,   187),
            DenseVector.toDenseVec(116,    54,   106,   143,   136,    81,   153),
            DenseVector.toDenseVec(130,    42,    95,   142,   136,    64,   158),
            DenseVector.toDenseVec(106,    79,    62,   114,    33,   123,    86),
            DenseVector.toDenseVec(149,    90,    99,   173,   103,   139,   152)
        } );
    }
    
    @AfterClass
    public static void tearDownClass()
    {
    }
    
    @Before
    public void setUp()
    {
    }
    
    @After
    public void tearDown()
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
        
        Matrix aCopy = A.clone();
        Matrix bCopy = B.clone();
        
        aCopy.mutableAdd(B);
        bCopy.mutableAdd(A);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        try
        {
            C.clone().mutableAdd(A);
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
        
        Matrix aCopy = A.clone();
        Matrix bCopy = B.clone();
        
        aCopy.mutableAdd(B, threadpool);
        bCopy.mutableAdd(A, threadpool);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        try
        {
            C.clone().mutableAdd(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testMutableAdd_double_Matrix_ExecutorService()
    {
        DenseMatrix ApB = new DenseMatrix(new double[][] 
        {
            {6,     8,     6,    16,    17},
            {2,    13,    10,     9,    15},
            {1,     5,    14,    10,    10},
            {6,    17,     5,    16,     6},
            {9,    12,     6,    12,     7}
        } );
        
        Matrix aCopy = A.clone();
        Matrix bCopy = B.clone();
        
        aCopy.mutableAdd(1.0, B, threadpool);
        bCopy.mutableAdd(1.0, A, threadpool);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        aCopy.mutableAdd(-1.0, B, threadpool);
        assertEquals(A, aCopy);
        
        Matrix Aadd5  = new DenseMatrix(A.rows(), A.cols());
        Aadd5.mutableAdd(5.0, A, threadpool);
        assertEquals(A.multiply(5), Aadd5);
        
        try
        {
            C.clone().mutableAdd(1.0, A, threadpool);
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
        
        Matrix aCopy = A.clone();
        
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
        
        Matrix aCopy = A.clone();
        
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
        
        Matrix aCopy = A.clone();
        Matrix bCopy = B.clone();
        
        aCopy.mutableSubtract(B);
        bCopy.mutableSubtract(A);
        
        assertEquals(AmB, aCopy);
        assertEquals(BmA, bCopy);
        
        try
        {
            C.clone().mutableSubtract(A);
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
        
        Matrix aCopy = A.clone();
        Matrix bCopy = B.clone();
        
        aCopy.mutableSubtract(B, threadpool);
        bCopy.mutableSubtract(A, threadpool);
        
        assertEquals(AmB, aCopy);
        assertEquals(BmA, bCopy);
        
        try
        {
            C.clone().mutableSubtract(A, threadpool);
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
        DenseVector b = new DenseVector(Arrays.asList(4.0, 5.0, 2.0, 6.0, 7.0));
        
        DenseVector z = new DenseVector(Arrays.asList(2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0));
        
        DenseVector Ab = new DenseVector(Arrays.asList(148.0, 110.0, 103.0, 94.0, 149.0));
        
        assertEquals(Ab, A.multiply(b));
        
        DenseVector Cz = new DenseVector(Arrays.asList(62.0, 100.0, 88.0, 74.0, 68.0));
        
        assertEquals(Cz, C.multiply(z));
    }
    
    /**
     * Test of multiply method, of class DenseMatrix.
     */
    @Test
    public void testMultiply_Vec_Double_Vec()
    {
        DenseVector b = new DenseVector(Arrays.asList(4.0, 5.0, 2.0, 6.0, 7.0));
        
        DenseVector z = new DenseVector(Arrays.asList(2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0));
                
        DenseVector store = b.deepCopy();
        
        A.multiply(b, 3.0, store);
        assertEquals(new DenseVector(new double[]{ 448, 335, 311, 288, 454}), store);
        
        DenseVector Cz = new DenseVector(Arrays.asList(62.0, 100.0, 88.0, 74.0, 68.0));
        
        store.zeroOut();
        C.multiply(z, 1.0, store);
        assertEquals(Cz, store);
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
    
    @Test
    public void testMultiply_Matrix_Matrix()
    {
        DenseMatrix R = new DenseMatrix(A.rows(), B.cols());
        
        A.multiply(B, R);
        assertEquals(AB, R);
        A.multiply(B, R);
        assertEquals(AB.multiply(2), R);
        
        R = new DenseMatrix(B.rows(), A.cols());
        B.multiply(A, R);
        assertEquals(BA, R);
        B.multiply(A, R);
        assertEquals(BA.multiply(2), R);
        
        R = new DenseMatrix(A.rows(), C.cols());
        A.multiply(C, R);
        assertEquals(AC, R);
        A.multiply(C, R);
        assertEquals(AC.multiply(2), R);
        
        try
        {
            R.multiply(A, C);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            A.multiply(B, C);
            fail("Expected error about target matrix dimensions"); 
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
    
    @Test
    public void testMultiply_Matrix_ExecutorService_Matrix()
    {
        DenseMatrix R = new DenseMatrix(A.rows(), B.cols());
        
        A.multiply(B, R, threadpool);
        assertEquals(AB, R);
        A.multiply(B, R, threadpool);
        assertEquals(AB.multiply(2), R);
        
        R = new DenseMatrix(B.rows(), A.cols());
        B.multiply(A, R, threadpool);
        assertEquals(BA, R);
        B.multiply(A, R, threadpool);
        assertEquals(BA.multiply(2), R);
        
        R = new DenseMatrix(A.rows(), C.cols());
        A.multiply(C, R, threadpool);
        assertEquals(AC, R);
        A.multiply(C, R, threadpool);
        assertEquals(AC.multiply(2), R);
        
        try
        {
            R.multiply(A, C, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            A.multiply(B, C, threadpool);
            fail("Expected error about target matrix dimensions"); 
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
        
        Matrix aCopy = A.clone();
        
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
        
        Matrix aCopy = A.clone();
        
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
     * Test of isSparse method, of class DenseMatrix.
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
     * Test of clone method, of class DenseMatrix.
     */
    @Test
    public void testCopy()
    {
        Matrix ACopy = A.clone();
        
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
        
        Matrix Expected = new DenseMatrix(new double[][] 
        {
            {5, 5, 3, 7, 2, 10, 0},
            {1, 2, 6, 5, 8, 1, 9},
            {8, 0, 5, 7, 9, 1, 8},
            {9, 3, 2, 7, 2, 4, 8},
            {1, 6, 8, 3, 1, 5, 10}
        } );
        
        Matrix test = C.clone();
        
        
        test.swapRows(1, 0);
        test.swapRows(1, 0);
        assertEquals(C, test);
        test.swapRows(0, 1);
        test.swapRows(0, 1);
        assertEquals(C, test);
        test.swapRows(3, 3);
        assertEquals(C, test);
        
        
        test.swapRows(0, 4);
        test.swapRows(0, 1);
        assertEquals(Expected, test);
        
        
        test = C.clone();
        test.swapRows(4, 0);
        test.swapRows(1, 0);
        assertEquals(Expected, test);
    }

    /**
     * Test of zeroOut method, of class DenseMatrix.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        
        Matrix test = C.clone();
        test.zeroOut();
        
        for(int i = 0; i < test.rows(); i++)
            for(int j = 0; j < test.cols(); j++)
                assertEquals(0, test.get(i, j), 0);
    }

    /**
     * Test of lup method, of class DenseMatrix.
     */
    @Test
    public void testLup_0args()
    {
        System.out.println("lup");
        
        Matrix[] lup;
        
        lup = A.clone().lup();
        assertTrue(lup[2].multiply(A).equals(lup[0].multiply(lup[1]), 1e-14));
        
        lup = C.clone().lup();
        assertTrue(lup[2].multiply(C).equals(lup[0].multiply(lup[1]), 1e-14));
        
        
        lup = C.transpose().lup();
        assertTrue(lup[2].multiply(C.transpose()).equals(lup[0].multiply(lup[1]), 1e-14));
    }

    /**
     * Test of lup method, of class DenseMatrix.
     */
    @Test
    public void testLup_ExecutorService()
    {
        System.out.println("lup");
        
        Matrix[] lup;
        
        lup = A.clone().lup(threadpool);
        assertTrue(lup[2].multiply(A, threadpool).equals(lup[0].multiply(lup[1], threadpool), 1e-14));
        
        lup = C.clone().lup(threadpool);
        assertTrue(lup[2].multiply(C, threadpool).equals(lup[0].multiply(lup[1], threadpool), 1e-14));
        
        
        lup = C.transpose().lup(threadpool);
        assertTrue(lup[2].multiply(C.transpose(), threadpool).equals(lup[0].multiply(lup[1], threadpool), 1e-14));
    }

    /**
     * Test of mutableTranspose method, of class DenseMatrix.
     */
    @Test
    public void testMutableTranspose()
    {
        System.out.println("mutableTranspose");
        
        try
        {
            C.clone().mutableTranspose();
            fail("Can not do a mutable transpose for rectangular matrix, error should have been thrown");
        }
        catch(Exception ex)
        {
            
        }
        
        
        DenseMatrix ATranspose = new DenseMatrix(new double[][] 
        {
            {1,     1,     0,     3,     1},
            {5,     5,     3,     8,     9},
            {4,     7,     8,     0,     2},
            {8,     3,     5,     7,     9},
            {9,     7,     6,     0,     6}, 
        } );
        
        Matrix AT = A.clone();
        AT.mutableTranspose();
        assertEquals(ATranspose, AT);
        
    }

    /**
     * Test of qr method, of class DenseMatrix.
     */
    @Test
    public void testQr_0args()
    {
        System.out.println("qr");
        Matrix[] qr;
        //3 properties to test
        // R is uper triangular
        // Q*Q' = I
        // A = Q*R
        
        
        
        qr = A.clone().qr();
        assertTrue(A.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(A.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = B.clone().qr();
        assertTrue(B.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(B.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = C.clone().qr();
        assertTrue(C.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(C.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        qr = C.transpose().qr();
        assertTrue(C.transpose().equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(C.transpose().rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
    }

    /**
     * Test of qr method, of class DenseMatrix.
     */
    @Test
    public void testQr_ExecutorService()
    {
        System.out.println("qr");
         Matrix[] qr;
        //3 properties to test
        // R is uper triangular
        // Q*Q' = I
        // A = Q*R
        
        
        
        qr = A.clone().qr(threadpool);
        assertTrue(A.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(A.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = B.clone().qr(threadpool);
        assertTrue(B.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(B.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = C.clone().qr(threadpool);
        assertTrue(C.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(C.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        qr = C.transpose().qr(threadpool);
        assertTrue(C.transpose().equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(DenseMatrix.eye(C.transpose().rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
    }
    
    
    @Test
    public void testTransposeMultiply_Matrix()
    {
        Matrix result;
        
        result = A.transpose().transposeMultiply(B);
        assertEquals(AB, result);
        
        result = B.transpose().transposeMultiply(A);
        assertEquals(BA, result);
        
        result = A.transpose().transposeMultiply(C);
        assertEquals(AC, result); 
        
        result = C.transposeMultiply(A);
        assertEquals(new DenseMatrix(new double[][] 
        { 
            {34,   135,   105,   135,    98},
            {22,    97,    63,   102,   101},
            {23,   140,   105,   166,   159},
            {36,   172,   127,   174,   148},
            {17,   130,   106,   145,   125},
            {28,   119,   100,   112,   127},
            {43,   219,   122,   257,   192},
        } ), result); 
        
        result = C.transposeMultiply(C);
        assertEquals(new DenseMatrix(new double[][]
        {
            {172,    60,    87,   162,   109,   100,   155},
            { 60,    74,    81,    84,    38,    94,   102},
            { 87,    81,   138,   124,   111,    89,   190},
            {162,    84,   124,   181,   134,   125,   187},
            {109,    38,   111,   134,   154,    50,   170},
            {100,    94,    89,   125,    50,   143,    99},
            {155,   102,   190,   187,   170,    99,   309},
        }), result); 
        
        try
        {
            C.transpose().transposeMultiply(A);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testTransposeMultiply_Matrix_Matrix()
    {
        Matrix R = new DenseMatrix(A.rows(), B.cols());
        
        A.transpose().transposeMultiply(B, R);
        assertEquals(AB, R);
        A.transpose().transposeMultiply(B, R);
        assertEquals(AB.multiply(2), R);
        
        R = new DenseMatrix(B.rows(), A.cols());
        B.transpose().transposeMultiply(A, R);
        assertEquals(BA, R);
        B.transpose().transposeMultiply(A, R);
        assertEquals(BA.multiply(2), R);
        
        R = new DenseMatrix(A.rows(), C.cols());
        A.transpose().transposeMultiply(C, R);
        assertEquals(AC, R); 
        A.transpose().transposeMultiply(C, R);
        assertEquals(AC.multiply(2), R); 
        
        R = new DenseMatrix(C.cols(), A.cols());
        C.transposeMultiply(A, R);
        Matrix CtA = new DenseMatrix(new double[][] 
        { 
            {34,   135,   105,   135,    98},
            {22,    97,    63,   102,   101},
            {23,   140,   105,   166,   159},
            {36,   172,   127,   174,   148},
            {17,   130,   106,   145,   125},
            {28,   119,   100,   112,   127},
            {43,   219,   122,   257,   192},
        } );
        assertEquals(CtA, R); 
        C.transposeMultiply(A, R);
        assertEquals(CtA.multiply(2), R);
        
        R = new DenseMatrix(C.cols(), C.cols());
        C.transposeMultiply(C, R);
        Matrix CtC = new DenseMatrix(new double[][]
        {
            {172,    60,    87,   162,   109,   100,   155},
            { 60,    74,    81,    84,    38,    94,   102},
            { 87,    81,   138,   124,   111,    89,   190},
            {162,    84,   124,   181,   134,   125,   187},
            {109,    38,   111,   134,   154,    50,   170},
            {100,    94,    89,   125,    50,   143,    99},
            {155,   102,   190,   187,   170,    99,   309},
        });
        assertEquals(CtC, R); 
        C.transposeMultiply(C, R);
        assertEquals(CtC.multiply(2), R); 
        
        try
        {
            A.transpose().transposeMultiply(B, R);
            fail("Expected error about target matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            C.transpose().transposeMultiply(A, R);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    
    @Test
    public void testTransposeMultiply_Matrix_ExecutorService()
    {
        Matrix result;
        
        result = A.transpose().transposeMultiply(B, threadpool);
        assertEquals(AB, result);
        
        result = B.transpose().transposeMultiply(A, threadpool);
        assertEquals(BA, result);
        
        result = A.transpose().transposeMultiply(C, threadpool);
        assertEquals(AC, result); 
        
        result = C.transposeMultiply(A, threadpool);
        assertEquals(new DenseMatrix(new double[][] 
        { 
            {34,   135,   105,   135,    98},
            {22,    97,    63,   102,   101},
            {23,   140,   105,   166,   159},
            {36,   172,   127,   174,   148},
            {17,   130,   106,   145,   125},
            {28,   119,   100,   112,   127},
            {43,   219,   122,   257,   192},
        } ), result); 
        
        result = C.transposeMultiply(C, threadpool);
        assertEquals(new DenseMatrix(new double[][]
        {
            {172,    60,    87,   162,   109,   100,   155},
            { 60,    74,    81,    84,    38,    94,   102},
            { 87,    81,   138,   124,   111,    89,   190},
            {162,    84,   124,   181,   134,   125,   187},
            {109,    38,   111,   134,   154,    50,   170},
            {100,    94,    89,   125,    50,   143,    99},
            {155,   102,   190,   187,   170,    99,   309},
        }), result); 
        
        try
        {
            C.transpose().transposeMultiply(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testTransposeMultiply_Matrix_Matrix_ExecutorService()
    {
        Matrix R = new DenseMatrix(A.rows(), B.cols());
        
        A.transpose().transposeMultiply(B, R, threadpool);
        assertEquals(AB, R);
        A.transpose().transposeMultiply(B, R, threadpool);
        assertEquals(AB.multiply(2), R);
        
        R = new DenseMatrix(B.rows(), A.cols());
        B.transpose().transposeMultiply(A, R, threadpool);
        assertEquals(BA, R);
        B.transpose().transposeMultiply(A, R, threadpool);
        assertEquals(BA.multiply(2), R);
        
        R = new DenseMatrix(A.rows(), C.cols());
        A.transpose().transposeMultiply(C, R, threadpool);
        assertEquals(AC, R); 
        A.transpose().transposeMultiply(C, R, threadpool);
        assertEquals(AC.multiply(2), R); 
        
        R = new DenseMatrix(C.cols(), A.cols());
        C.transposeMultiply(A, R, threadpool);
        Matrix CtA = new DenseMatrix(new double[][] 
        { 
            {34,   135,   105,   135,    98},
            {22,    97,    63,   102,   101},
            {23,   140,   105,   166,   159},
            {36,   172,   127,   174,   148},
            {17,   130,   106,   145,   125},
            {28,   119,   100,   112,   127},
            {43,   219,   122,   257,   192},
        } );
        assertEquals(CtA, R); 
        C.transposeMultiply(A, R, threadpool);
        assertEquals(CtA.multiply(2), R);
        
        R = new DenseMatrix(C.cols(), C.cols());
        C.transposeMultiply(C, R, threadpool);
        Matrix CtC = new DenseMatrix(new double[][]
        {
            {172,    60,    87,   162,   109,   100,   155},
            { 60,    74,    81,    84,    38,    94,   102},
            { 87,    81,   138,   124,   111,    89,   190},
            {162,    84,   124,   181,   134,   125,   187},
            {109,    38,   111,   134,   154,    50,   170},
            {100,    94,    89,   125,    50,   143,    99},
            {155,   102,   190,   187,   170,    99,   309},
        });
        assertEquals(CtC, R); 
        C.transposeMultiply(C, R, threadpool);
        assertEquals(CtC.multiply(2), R); 
        
        try
        {
            A.transpose().transposeMultiply(B, R, threadpool);
            fail("Expected error about target matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            C.transpose().transposeMultiply(A, R, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testTransposeMultiply_Double_Vec()
    {
        DenseVector b = new DenseVector(Arrays.asList(4.0, 5.0, 2.0, 6.0, 7.0));
        
        DenseVector z = new DenseVector(Arrays.asList(2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0));
        
        DenseVector Ab = new DenseVector(Arrays.asList(148.0, 110.0, 103.0, 94.0, 149.0));
        
        assertEquals(Ab, A.transpose().transposeMultiply(1.0, b));
        
        assertEquals(Ab.multiply(7.0), A.transpose().transposeMultiply(7.0, b));
        
        DenseVector Cz = new DenseVector(Arrays.asList(62.0, 100.0, 88.0, 74.0, 68.0));
        
        assertEquals(Cz, C.transpose().transposeMultiply(1.0, z));
        
        assertEquals(Cz.multiply(19.0), C.transpose().transposeMultiply(19.0, z));
        
        try
        {
            C.transposeMultiply(1.0, z);
            fail("Dimensions were in disagreement, should not have worked");
        }
        catch(Exception ex)
        {
            
        }
    }
    
    @Test
    public void testChangeSize()
    {
        MatrixOfVecs Acpy = A.clone();
        Acpy.changeSize(Acpy.rows()-1, Acpy.cols()-1);
        assertEquals(Acpy.rows(), A.rows()-1);
        assertEquals(Acpy.cols(), A.cols()-1);
        
        
        for(int i = 0; i < Acpy.rows(); i++)
            for(int j = 0; j < Acpy.cols(); j++)
                assertEquals(Acpy.get(i, j), A.get(i, j), 0.0);
        //Expand back out and make sure the values are zero on the sides
        Acpy.changeSize(Acpy.rows()+2, Acpy.cols()+2);
        assertEquals(Acpy.rows(), A.rows()+1);
        assertEquals(Acpy.cols(), A.cols()+1);
        
        for(int i = 0; i < Acpy.rows(); i++)
            for(int j = 0; j < Acpy.cols(); j++)
                if(i < A.rows()-1 && j < A.cols()-1)
                    assertEquals(A.get(i, j), Acpy.get(i, j), 0.0);
                else
                    assertEquals(0.0, Acpy.get(i, j), 0.0);
    }
}

