package jsat.linear;

import java.util.Arrays;
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
public class GenericMatrixTest
{
    protected static class TestImp extends GenericMatrix
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = 1487285165522642650L;
		private final double[][] storage;

        public TestImp(final double[][] storage)
        {
            this.storage = storage;
        }
        
        public TestImp(final int rows, final int cols)
        {
            storage = new double[rows][cols];
        }
        
        @Override
        protected Matrix getMatrixOfSameType(final int rows, final int cols)
        {
            return new TestImp(rows, cols);
        }

        @Override
        public double get(final int i, final int j)
        {
            return storage[i][j];
        }

        @Override
        public void set(final int i, final int j, final double value)
        {
            storage[i][j] = value;
        }

        @Override
        public int rows()
        {
            return storage.length;
        }

        @Override
        public int cols()
        {
            return storage[0].length;
        }

        @Override
        public boolean isSparce()
        {
            return false;
        }

        @Override
        public void changeSize(final int newRows, final int newCols)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
        
    }
    
    /**
     * 5x5
     */
    static TestImp A;
    /**
     * 5x5
     */
    static TestImp B;
    /**
     * 5x7
     */
    static TestImp C;
    
    /**
     * 7x5
     */
    static TestImp D;
    
    static TestImp AB;
    static TestImp BA;
    static TestImp AC;
    
    /**
     * Multi threaded pool with daemon threads
     */
    static ExecutorService threadpool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()+1, new ThreadFactory() {

        public Thread newThread(final Runnable r)
        {
            final Thread thread = new Thread(r);
            thread.setDaemon(true);
            return thread;
        }
    });
    
    public GenericMatrixTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        A = new TestImp(new double[][] 
        {
            {1, 5, 4, 8, 9},
            {1, 5, 7, 3, 7},
            {0, 3, 8, 5, 6},
            {3, 8, 0, 7, 0},
            {1, 9, 2, 9, 6}
        } );
        
        B = new TestImp(new double[][] 
        {
            {5, 3, 2, 8, 8},
            {1, 8, 3, 6, 8},
            {1, 2, 6, 5, 4},
            {3, 9, 5, 9, 6},
            {8, 3, 4, 3, 1}
        } );
        
        C = new TestImp(new double[][] 
        {
            {1, 6, 8, 3, 1, 5, 10},
            {5, 5, 3, 7, 2, 10, 0},
            {8, 0, 5, 7, 9, 1, 8},
            {9, 3, 2, 7, 2, 4, 8},
            {1, 2, 6, 5, 8, 1, 9}
        } );
        
        D = new TestImp(new double[][] 
        {
            { 2,    4,    5,    4,    4},
            {10,    3,    2,    0,    7},
            { 4,    5,    1,    7,    7},
            { 2,    7,    2,    4,    7},
            { 6,    2,    9,    2,    4},
            { 1,    5,    6,    5,   10},
            { 3,    4,    1,    5,    0},
        } );
        
        AB = new TestImp(new double[][] 
        {
            {110,   150,   117,   157,   121},
            {82,   105,   102,   121,   101},
            {74,   103,   106,   121,    92},
            {44,   136,    65,   135,   130},
            {91,   178,   110,   171,   148}
        } );
        
        BA = new TestImp(new double[][] 
        {
            {40,   182,    73,   187,   126},
            {35,   174,   100,   161,   131},
            {22,   109,    74,   115,    83},
            {45,   201,   127,   193,   156},
            {21,   100,    87,   123,   123}
        } );
        
        AC = new TestImp(new double[][] 
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
     * Test of mutableAdd method, of class TestImp.
     */
    @Test
    public void testMutableAdd_Matrix()
    {
        final TestImp ApB = new TestImp(new double[][] 
        {
            {6,     8,     6,    16,    17},
            {2,    13,    10,     9,    15},
            {1,     5,    14,    10,    10},
            {6,    17,     5,    16,     6},
            {9,    12,     6,    12,     7}
        } );
        
        final Matrix aCopy = A.clone();
        final Matrix bCopy = B.clone();
        
        aCopy.mutableAdd(B);
        bCopy.mutableAdd(A);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        try
        {
            C.clone().mutableAdd(A);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableAdd method, of class TestImp.
     */
    @Test
    public void testMutableAdd_Matrix_ExecutorService()
    {
        final TestImp ApB = new TestImp(new double[][] 
        {
            {6,     8,     6,    16,    17},
            {2,    13,    10,     9,    15},
            {1,     5,    14,    10,    10},
            {6,    17,     5,    16,     6},
            {9,    12,     6,    12,     7}
        } );
        
        final Matrix aCopy = A.clone();
        final Matrix bCopy = B.clone();
        
        aCopy.mutableAdd(B, threadpool);
        bCopy.mutableAdd(A, threadpool);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        try
        {
            C.clone().mutableAdd(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testMutableAdd_double_Matrix_ExecutorService()
    {
        final TestImp ApB = new TestImp(new double[][] 
        {
            {6,     8,     6,    16,    17},
            {2,    13,    10,     9,    15},
            {1,     5,    14,    10,    10},
            {6,    17,     5,    16,     6},
            {9,    12,     6,    12,     7}
        } );
        
        final Matrix aCopy = A.clone();
        final Matrix bCopy = B.clone();
        
        aCopy.mutableAdd(1.0, B, threadpool);
        bCopy.mutableAdd(1.0, A, threadpool);
        
        assertEquals(ApB, aCopy);
        assertEquals(ApB, bCopy);
        
        aCopy.mutableAdd(-1.0, B, threadpool);
        assertEquals(A, aCopy);
        
        final Matrix Aadd5  = new TestImp(A.rows(), A.cols());
        Aadd5.mutableAdd(5.0, A, threadpool);
        assertEquals(A.multiply(5), Aadd5);
        
        try
        {
            C.clone().mutableAdd(1.0, A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableAdd method, of class TestImp.
     */
    @Test
    public void testMutableAdd_double()
    {
        final TestImp ApTwo = new TestImp(new double[][] 
        {
            {1+2, 5+2, 4+2, 8+2, 9+2},
            {1+2, 5+2, 7+2, 3+2, 7+2},
            {0+2, 3+2, 8+2, 5+2, 6+2},
            {3+2, 8+2, 0+2, 7+2, 0+2},
            {1+2, 9+2, 2+2, 9+2, 6+2}
        } );
        
        final Matrix aCopy = A.clone();
        
        aCopy.mutableAdd(2);
        
        assertEquals(ApTwo, aCopy);
    }

    /**
     * Test of mutableAdd method, of class TestImp.
     */
    @Test
    public void testMutableAdd_double_ExecutorService()
    {
        final TestImp ApTwo = new TestImp(new double[][] 
        {
            {1+2, 5+2, 4+2, 8+2, 9+2},
            {1+2, 5+2, 7+2, 3+2, 7+2},
            {0+2, 3+2, 8+2, 5+2, 6+2},
            {3+2, 8+2, 0+2, 7+2, 0+2},
            {1+2, 9+2, 2+2, 9+2, 6+2}
        } );
        
        final Matrix aCopy = A.clone();
        
        aCopy.mutableAdd(2, threadpool);
        
        assertEquals(ApTwo, aCopy);
    }

    /**
     * Test of mutableSubtract method, of class TestImp.
     */
    @Test
    public void testMutableSubtract_Matrix()
    {
        final TestImp AmB = new TestImp(new double[][] 
        {
            {-4,     2,     2,     0,     1},
            { 0,    -3,     4,    -3,    -1},
            {-1,     1,     2,     0,     2},
            { 0,    -1,    -5,    -2,    -6},
            {-7,     6,    -2,     6,     5}
        } );
        
        final TestImp BmA = new TestImp(new double[][] 
        {
            {-4*-1,     2*-1,     2*-1,     0*-1,     1*-1},
            { 0*-1,    -3*-1,     4*-1,    -3*-1,    -1*-1},
            {-1*-1,     1*-1,     2*-1,     0*-1,     2*-1},
            { 0*-1,    -1*-1,    -5*-1,    -2*-1,    -6*-1},
            {-7*-1,     6*-1,    -2*-1,     6*-1,     5*-1}
        } );
        
        final Matrix aCopy = A.clone();
        final Matrix bCopy = B.clone();
        
        aCopy.mutableSubtract(B);
        bCopy.mutableSubtract(A);
        
        assertEquals(AmB, aCopy);
        assertEquals(BmA, bCopy);
        
        try
        {
            C.clone().mutableSubtract(A);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableSubtract method, of class TestImp.
     */
    @Test
    public void testMutableSubtract_Matrix_ExecutorService()
    {
        final TestImp AmB = new TestImp(new double[][] 
        {
            {-4,     2,     2,     0,     1},
            { 0,    -3,     4,    -3,    -1},
            {-1,     1,     2,     0,     2},
            { 0,    -1,    -5,    -2,    -6},
            {-7,     6,    -2,     6,     5}
        } );
        
        final TestImp BmA = new TestImp(new double[][] 
        {
            {-4*-1,     2*-1,     2*-1,     0*-1,     1*-1},
            { 0*-1,    -3*-1,     4*-1,    -3*-1,    -1*-1},
            {-1*-1,     1*-1,     2*-1,     0*-1,     2*-1},
            { 0*-1,    -1*-1,    -5*-1,    -2*-1,    -6*-1},
            {-7*-1,     6*-1,    -2*-1,     6*-1,     5*-1}
        } );
        
        final Matrix aCopy = A.clone();
        final Matrix bCopy = B.clone();
        
        aCopy.mutableSubtract(B, threadpool);
        bCopy.mutableSubtract(A, threadpool);
        
        assertEquals(AmB, aCopy);
        assertEquals(BmA, bCopy);
        
        try
        {
            C.clone().mutableSubtract(A, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of multiply method, of class TestImp.
     */
    @Test
    public void testMultiply_Vec()
    {
        final DenseVector b = new DenseVector(Arrays.asList(4.0, 5.0, 2.0, 6.0, 7.0));
        
        final DenseVector z = new DenseVector(Arrays.asList(2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0));
        
        final DenseVector Ab = new DenseVector(Arrays.asList(148.0, 110.0, 103.0, 94.0, 149.0));
        
        assertEquals(Ab, A.multiply(b));
        
        final DenseVector Cz = new DenseVector(Arrays.asList(62.0, 100.0, 88.0, 74.0, 68.0));
        
        assertEquals(Cz, C.multiply(z));
    }
    
    /**
     * Test of multiply method, of class TestImp.
     */
    @Test
    public void testMultiply_Vec_Double_Vec()
    {
        final DenseVector b = new DenseVector(Arrays.asList(4.0, 5.0, 2.0, 6.0, 7.0));
        
        final DenseVector z = new DenseVector(Arrays.asList(2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0));
                
        final DenseVector store = b.deepCopy();
        
        A.multiply(b, 3.0, store);
        assertEquals(new DenseVector(new double[]{ 448, 335, 311, 288, 454}), store);
        
        final DenseVector Cz = new DenseVector(Arrays.asList(62.0, 100.0, 88.0, 74.0, 68.0));
        
        store.zeroOut();
        C.multiply(z, 1.0, store);
        assertEquals(Cz, store);
    }
    
    /**
     * Test of multiply method, of class TestImp.
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
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testMultiplyTranspose_Matrix_Matrix()
    {
        final TestImp AmBt = new TestImp(new double[][] 
        {
            {164,    173,    111,   194,    72},
            {114,   136,    96 ,  152,    67},
            {113,   126,   103,   148,    62},
            { 95,   109,    54,   144,    69},
            {156,   181,   100,   211,    76}
        } );
        final TestImp DmBt = new TestImp(new double[][] 
        {
            { 96,   105,    76,   127,    64},
            {119,    96,    56,   109,   104},
            {149,   145,    83,   167,    79},
            {123,   144,    76,   157,    64},
            {102,    93,    90,   123,   100},
            {152,   169,   112,   183,    72},
            { 69,    68,    42,    95,    55},
        } );
        
        TestImp tmp = new TestImp(5, 5);
        A.multiplyTranspose(B, tmp);
        assertEquals(AmBt, tmp);
        
        tmp = new TestImp(7, 5);
        D.multiplyTranspose(B, tmp);
        assertEquals(DmBt, tmp);
    }
    
    @Test
    public void testMultiplyTranspose_Matrix_Matrix_Exector()
    {
        final TestImp AmBt = new TestImp(new double[][] 
        {
            {164,    173,    111,   194,    72},
            {114,   136,    96 ,  152,    67},
            {113,   126,   103,   148,    62},
            { 95,   109,    54,   144,    69},
            {156,   181,   100,   211,    76}
        } );
        final TestImp DmBt = new TestImp(new double[][] 
        {
            { 96,   105,    76,   127,    64},
            {119,    96,    56,   109,   104},
            {149,   145,    83,   167,    79},
            {123,   144,    76,   157,    64},
            {102,    93,    90,   123,   100},
            {152,   169,   112,   183,    72},
            { 69,    68,    42,    95,    55},
        } );
        
        TestImp tmp = new TestImp(5, 5);
        A.multiplyTranspose(B, tmp, threadpool);
        assertEquals(AmBt, tmp);
        
        tmp = new TestImp(7, 5);
        D.multiplyTranspose(B, tmp, threadpool);
        assertEquals(DmBt, tmp);
    }
    
    @Test
    public void testMultiply_Matrix_Matrix()
    {
        TestImp R = new TestImp(A.rows(), B.cols());
        
        A.multiply(B, R);
        assertEquals(AB, R);
        A.multiply(B, R);
        assertEquals(AB.multiply(2), R);
        
        R = new TestImp(B.rows(), A.cols());
        B.multiply(A, R);
        assertEquals(BA, R);
        B.multiply(A, R);
        assertEquals(BA.multiply(2), R);
        
        R = new TestImp(A.rows(), C.cols());
        A.multiply(C, R);
        assertEquals(AC, R);
        A.multiply(C, R);
        assertEquals(AC.multiply(2), R);
        
        try
        {
            R.multiply(A, C);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            A.multiply(B, C);
            fail("Expected error about target matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of multiply method, of class TestImp.
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
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testMultiply_Matrix_ExecutorService_Matrix()
    {
        TestImp R = new TestImp(A.rows(), B.cols());
        
        A.multiply(B, R, threadpool);
        assertEquals(AB, R);
        A.multiply(B, R, threadpool);
        assertEquals(AB.multiply(2), R);
        
        R = new TestImp(B.rows(), A.cols());
        B.multiply(A, R, threadpool);
        assertEquals(BA, R);
        B.multiply(A, R, threadpool);
        assertEquals(BA.multiply(2), R);
        
        R = new TestImp(A.rows(), C.cols());
        A.multiply(C, R, threadpool);
        assertEquals(AC, R);
        A.multiply(C, R, threadpool);
        assertEquals(AC.multiply(2), R);
        
        try
        {
            R.multiply(A, C, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            A.multiply(B, C, threadpool);
            fail("Expected error about target matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }

    /**
     * Test of mutableMultiply method, of class TestImp.
     */
    @Test
    public void testMutableMultiply_double()
    {
        final TestImp AtTwo = new TestImp(new double[][] 
        {
            {1*2, 5*2, 4*2, 8*2, 9*2},
            {1*2, 5*2, 7*2, 3*2, 7*2},
            {0*2, 3*2, 8*2, 5*2, 6*2},
            {3*2, 8*2, 0*2, 7*2, 0*2},
            {1*2, 9*2, 2*2, 9*2, 6*2}
        } );
        
        final Matrix aCopy = A.clone();
        
        aCopy.mutableMultiply(2);
        
        assertEquals(AtTwo, aCopy);
    }

    /**
     * Test of mutableMultiply method, of class TestImp.
     */
    @Test
    public void testMutableMultiply_double_ExecutorService()
    {
        final TestImp AtTwo = new TestImp(new double[][] 
        {
            {1*2, 5*2, 4*2, 8*2, 9*2},
            {1*2, 5*2, 7*2, 3*2, 7*2},
            {0*2, 3*2, 8*2, 5*2, 6*2},
            {3*2, 8*2, 0*2, 7*2, 0*2},
            {1*2, 9*2, 2*2, 9*2, 6*2}
        } );
        
        final Matrix aCopy = A.clone();
        
        aCopy.mutableMultiply(2, threadpool);
        
        assertEquals(AtTwo, aCopy);
    }

    /**
     * Test of transpose method, of class TestImp.
     */
    @Test
    public void testTranspose()
    {
        final TestImp CTranspose = new TestImp(new double[][] 
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
     * Test of get method, of class TestImp.
     */
    @Test
    public void testGet()
    {
        //Tests both
        testSet();
    }

    /**
     * Test of set method, of class TestImp.
     */
    @Test
    public void testSet()
    {
        final TestImp toSet = new TestImp(A.rows(), A.cols());
        
        for(int i = 0; i < A.rows(); i++) {
          for (int j = 0; j < A.cols(); j++) {
            toSet.set(i, j, A.get(i, j));
          }
        }
        
        assertEquals(A, toSet);
    }

    /**
     * Test of rows method, of class TestImp.
     */
    @Test
    public void testRows()
    {
        assertEquals(5, A.rows());
    }

    /**
     * Test of cols method, of class TestImp.
     */
    @Test
    public void testCols()
    {
        assertEquals(5, A.cols());
        assertEquals(7, C.cols());
    }

    /**
     * Test of isSparse method, of class TestImp.
     */
    @Test
    public void testIsSparce()
    {
        assertEquals(false, A.isSparce());
    }

    /**
     * Test of nnz method, of class TestImp.
     */
    @Test
    public void testNnz()
    {
        assertEquals(5*5, A.nnz());
        assertEquals(5*7, C.nnz());
    }

    /**
     * Test of clone method, of class TestImp.
     */
    @Test
    public void testCopy()
    {
        final Matrix ACopy = A.clone();
        
        assertEquals(A, ACopy);
        assertEquals(A.multiply(B), ACopy.multiply(B));
    }

    /**
     * Test of swapRows method, of class TestImp.
     */
    @Test
    public void testSwapRows()
    {
        System.out.println("swapRows");
        
        final Matrix Expected = new TestImp(new double[][] 
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
     * Test of zeroOut method, of class TestImp.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        
        final Matrix test = C.clone();
        test.zeroOut();
        
        for(int i = 0; i < test.rows(); i++) {
          for (int j = 0; j < test.cols(); j++) {
            assertEquals(0, test.get(i, j), 0);
          }
        }
    }

    /**
     * Test of lup method, of class TestImp.
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
     * Test of lup method, of class TestImp.
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
     * Test of mutableTranspose method, of class TestImp.
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
        catch(final Exception ex)
        {
            
        }
        
        
        final TestImp ATranspose = new TestImp(new double[][] 
        {
            {1,     1,     0,     3,     1},
            {5,     5,     3,     8,     9},
            {4,     7,     8,     0,     2},
            {8,     3,     5,     7,     9},
            {9,     7,     6,     0,     6}, 
        } );
        
        final Matrix AT = A.clone();
        AT.mutableTranspose();
        assertEquals(ATranspose, AT);
        
    }

    /**
     * Test of qr method, of class TestImp.
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
        assertTrue(Matrix.eye(A.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = B.clone().qr();
        assertTrue(B.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(Matrix.eye(B.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = C.clone().qr();
        assertTrue(C.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(Matrix.eye(C.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        qr = C.transpose().qr();
        assertTrue(C.transpose().equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(Matrix.eye(C.transpose().rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
    }

    /**
     * Test of qr method, of class TestImp.
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
        assertTrue(Matrix.eye(A.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = B.clone().qr(threadpool);
        assertTrue(B.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(Matrix.eye(B.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        
        qr = C.clone().qr(threadpool);
        assertTrue(C.equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(Matrix.eye(C.rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
        
        qr = C.transpose().qr(threadpool);
        assertTrue(C.transpose().equals(qr[0].multiply(qr[1]), 1e-14));
        assertTrue(Matrix.eye(C.transpose().rows()).equals(qr[0].multiply(qr[0].transpose()), 1e-14));
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
        assertEquals(new TestImp(new double[][] 
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
        assertEquals(new TestImp(new double[][]
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
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testTransposeMultiply_Matrix_Matrix()
    {
        Matrix R = new TestImp(A.rows(), B.cols());
        
        A.transpose().transposeMultiply(B, R);
        assertEquals(AB, R);
        A.transpose().transposeMultiply(B, R);
        assertEquals(AB.multiply(2), R);
        
        R = new TestImp(B.rows(), A.cols());
        B.transpose().transposeMultiply(A, R);
        assertEquals(BA, R);
        B.transpose().transposeMultiply(A, R);
        assertEquals(BA.multiply(2), R);
        
        R = new TestImp(A.rows(), C.cols());
        A.transpose().transposeMultiply(C, R);
        assertEquals(AC, R); 
        A.transpose().transposeMultiply(C, R);
        assertEquals(AC.multiply(2), R); 
        
        R = new TestImp(C.cols(), A.cols());
        C.transposeMultiply(A, R);
        final Matrix CtA = new TestImp(new double[][] 
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
        
        R = new TestImp(C.cols(), C.cols());
        C.transposeMultiply(C, R);
        final Matrix CtC = new TestImp(new double[][]
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
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            C.transpose().transposeMultiply(A, R);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
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
        assertEquals(new TestImp(new double[][] 
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
        assertEquals(new TestImp(new double[][]
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
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testTransposeMultiply_Matrix_Matrix_ExecutorService()
    {
        Matrix R = new TestImp(A.rows(), B.cols());
        
        A.transpose().transposeMultiply(B, R, threadpool);
        assertEquals(AB, R);
        A.transpose().transposeMultiply(B, R, threadpool);
        assertEquals(AB.multiply(2), R);
        
        R = new TestImp(B.rows(), A.cols());
        B.transpose().transposeMultiply(A, R, threadpool);
        assertEquals(BA, R);
        B.transpose().transposeMultiply(A, R, threadpool);
        assertEquals(BA.multiply(2), R);
        
        R = new TestImp(A.rows(), C.cols());
        A.transpose().transposeMultiply(C, R, threadpool);
        assertEquals(AC, R); 
        A.transpose().transposeMultiply(C, R, threadpool);
        assertEquals(AC.multiply(2), R); 
        
        R = new TestImp(C.cols(), A.cols());
        C.transposeMultiply(A, R, threadpool);
        final Matrix CtA = new TestImp(new double[][] 
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
        
        R = new TestImp(C.cols(), C.cols());
        C.transposeMultiply(C, R, threadpool);
        final Matrix CtC = new TestImp(new double[][]
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
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
        
        try
        {
            C.transpose().transposeMultiply(A, R, threadpool);
            fail("Expected error about matrix dimensions"); 
        }
        catch(final ArithmeticException ex)
        {
            //Good! We expected failure
        }
    }
    
    @Test
    public void testTransposeMultiply_Double_Vec()
    {
        final DenseVector b = new DenseVector(Arrays.asList(4.0, 5.0, 2.0, 6.0, 7.0));
        
        final DenseVector z = new DenseVector(Arrays.asList(2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0));
        
        final DenseVector Ab = new DenseVector(Arrays.asList(148.0, 110.0, 103.0, 94.0, 149.0));
        
        assertEquals(Ab, A.transpose().transposeMultiply(1.0, b));
        
        assertEquals(Ab.multiply(7.0), A.transpose().transposeMultiply(7.0, b));
        
        final DenseVector Cz = new DenseVector(Arrays.asList(62.0, 100.0, 88.0, 74.0, 68.0));
        
        assertEquals(Cz, C.transpose().transposeMultiply(1.0, z));
        
        assertEquals(Cz.multiply(19.0), C.transpose().transposeMultiply(19.0, z));
        
        try
        {
            C.transposeMultiply(1.0, z);
            fail("Dimensions were in disagreement, should not have worked");
        }
        catch(final Exception ex)
        {
            
        }
    }
}
