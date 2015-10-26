package jsat.linear;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
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
public class SparseMatrixTest
{
    /*
     * NOTE: True resultes computed with octave and stroed as row, column, value 
     * pairs. These are obtained using the find(X) comand. This also means the 
     * incicies are 1 based, so -1 will be used. 
     */
    private Matrix A;
    private Matrix B;
    private Matrix C;
    private Matrix Ct;
    
    private static ExecutorService ex;
    
    public SparseMatrixTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        ex.shutdown();
    }
    
    @Before
    public void setUp()
    {
        A = new SparseMatrix(6, 6, 2);
        A.set(0, 0, 1.0);
        A.set(0, 4, 4.0);
        A.set(1, 3, 3.0);
        A.set(2, 1, 2.0);
        A.set(2, 2, -7.0);
        A.set(3, 1, 1.0);
        A.set(4, 5, 1.0);
        A.set(5, 4, 3.0);
        
        B = new SparseMatrix(6, 6);
        B.set(0, 4, 2.0);
        B.set(1, 0, 3.0);
        B.set(1, 3, 2.0);
        B.set(2, 4, 5.0);
        B.set(2, 5, 1.0);
        B.set(3, 2, 3.0);
        B.set(3, 5, 4.0);
        B.set(4, 0, 2.0);
        B.set(5, 1, -2.0);
        B.set(5, 3, 1.0);
        
        
        C = new SparseMatrix(6, 8);
        Ct = new SparseMatrix(8, 6);
        C.set(0, 2, 1.0);
        C.set(0, 7, 1.0);
        C.set(1, 5, 1.0);
        C.set(2, 0, 1.0);
        C.set(2, 7, 1.0);
        C.set(3, 2, 1.0);
        C.set(3, 3, 1.0);
        C.set(3, 4, 1.0);
        C.set(4, 6, 1.0);
        C.set(5, 1, 1.0);
        
        Ct.set(2, 0, 1.0);
        Ct.set(7, 0, 1.0);
        Ct.set(5, 1, 1.0);
        Ct.set(0, 2, 1.0);
        Ct.set(7, 2, 1.0);
        Ct.set(2, 3, 1.0);
        Ct.set(3, 3, 1.0);
        Ct.set(4, 3, 1.0);
        Ct.set(6, 4, 1.0);
        Ct.set(1, 5, 1.0);
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of mutableAdd method, of class SparseMatrix.
     */
    @Test
    public void testMutableAdd_double_Matrix()
    {
        System.out.println("mutableAdd");

        final int[] r = new int[]
        {
            1, 2, 5, 3, 4, 6, 3, 4, 2, 6, 1, 3, 6, 3, 4, 5
        };

        final int[] c = new int[]
        {
            1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6
        };

        final double[] v = new double[]
        {
            1, 6, 4, 2, 1, -4, -7, 6, 7, 2, 8, 10, 3, 2, 8, 1
        };

        A.mutableAdd(2.0, B);
        
        checkAgainstRCV(A, v, r, c);
        
        try
        {
            A.mutableAdd(2, C);
            fail("Matrix dimensions do not agree");
        }
        catch (final ArithmeticException ex)
        {
        }
    }

    /**
     * Test of mutableAdd method, of class SparseMatrix.
     */
    @Test
    public void testMutableAdd_3args()
    {
        System.out.println("mutableAdd");
        final int[] r = new int[]
        {
            1, 2, 5, 3, 4, 6, 3, 4, 2, 6, 1, 3, 6, 3, 4, 5
        };

        final int[] c = new int[]
        {
            1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6
        };

        final double[] v = new double[]
        {
            1, 6, 4, 2, 1, -4, -7, 6, 7, 2, 8, 10, 3, 2, 8, 1
        };

        A.mutableAdd(2.0, B, ex);

        checkAgainstRCV(A, v, r, c);

        try
        {
            A.mutableAdd(2, C, ex);
            fail("Matrix dimensions do not agree");
        }
        catch (final ArithmeticException ex)
        {
        }
    }

    /**
     * Test of mutableAdd method, of class SparseMatrix.
     */
    @Test
    public void testMutableAdd_double()
    {
        System.out.println("mutableAdd");
        final DenseMatrix truth = new DenseMatrix(new double[][]
        {
            {2, 2, 2, 2, 4, 2},
            {5, 2, 2, 4, 2, 2},
            {2, 2, 2, 2, 7, 3},
            {2, 2, 5, 2, 2, 6},
            {4, 2, 2, 2, 2, 2},
            {2, 0, 2, 3, 2, 2},
        });
        
        B.mutableAdd(2);
        
        assertTrue(B.equals(truth, 1e-20));
        assertEquals(B.rows()*B.cols()-1, B.nnz());
    }

    /**
     * Test of mutableAdd method, of class SparseMatrix.
     */
    @Test
    public void testMutableAdd_double_ExecutorService()
    {
        System.out.println("mutableAdd");
        final DenseMatrix truth = new DenseMatrix(new double[][]
        {
            {2, 2, 2, 2, 4, 2},
            {5, 2, 2, 4, 2, 2},
            {2, 2, 2, 2, 7, 3},
            {2, 2, 5, 2, 2, 6},
            {4, 2, 2, 2, 2, 2},
            {2, 0, 2, 3, 2, 2},
        });
        
        B.mutableAdd(2, ex);
        
        assertTrue(B.equals(truth, 1e-20));
        assertEquals(B.rows()*B.cols()-1, B.nnz());
    }

    /**
     * Test of multiply method, of class SparseMatrix.
     */
    @Test
    public void testMultiply_3args_1()
    {
        System.out.println("multiply");
        final Vec b = new DenseVector(new double[]
        {
            5, 3, 3, 5, 4, 6
        });

        final Vec A2b = new DenseVector(new double[]
        {
            42, 30, -30, 6, 12, 24
        });
        final Vec B2b = new DenseVector(new double[]
        {
            16, 50, 52, 66, 20, -2
        });
        final Vec Ct2b = new DenseVector(new double[]
        {
            6, 12, 20, 10, 10, 6, 8, 16
        });
        
        DenseVector c = new DenseVector(6);
        
        c.zeroOut();
        A.multiply(b, 2, c);
        assertTrue(c.equals(A2b, 1e-20));
        
        c.zeroOut();
        B.multiply(b, 2, c);
        assertTrue(c.equals(B2b, 1e-20));
        
        try
        {
            c.zeroOut();
            C.multiply(b, 2, c);
            fail("Target vector does not agre, should have failed");
        }
        catch(final Exception ex)
        {
            
        }
        
        try
        {
            c.zeroOut();
            Ct.multiply(b, 2, c);
            fail("Target vector does not agre, should have failed");
        }
        catch(final Exception ex)
        {
            
        }
        
        c = new DenseVector(8);
        c.zeroOut();
        Ct.multiply(b, 2, c);
        assertTrue(c.equals(Ct2b, 1e-20));
        
    }

    /**
     * Test of multiply method, of class SparseMatrix.
     */
    @Test
    public void testMultiply_Matrix_Matrix()
    {
        System.out.println("multiply");

        SparseMatrix tmp;

        final int[] rAB = new int[]
        {
            1, 3, 4, 6, 5, 2, 3, 4, 5, 1, 3, 2, 3
        };

        final int[] cAB = new int[]
        {
            1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 6, 6
        };

        final double[] vAB = new double[]
        {
            8, 6, 3, 6, -2, 9, 4, 2, 1, 2, -35, 12, -7
        };

        tmp = new SparseMatrix(6, 6);
        A.multiply(B, tmp);
        checkAgainstRCV(tmp, vAB, rAB, cAB);



        final int[] rBA = new int[]
        {
            2, 5, 2, 4, 6, 4, 6, 2, 3, 4, 5, 1, 3
        };

        final int[] cBA = new int[]
        {
            1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 5, 6, 6
        };

        final double[] vBA = new double[]
        {
            3, 2, 2, 6, 1, -21, -6, 12, 3, 12, 8, 2, 5
        };

        tmp = new SparseMatrix(6, 6);
        B.multiply(A, tmp);
        checkAgainstRCV(tmp, vBA, rBA, cBA);

        final int[] rAC = new int[]
        {
            3, 5, 1, 2, 2, 2, 3, 4, 1, 6, 1, 3
        };

        final int[] cAC = new int[]
        {
            1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8
        };

        final double[] vAC = new double[]
        {
            -7, 1, 1, 3, 3, 3, 2, 1, 4, 3, 1, -7
        };

        tmp = new SparseMatrix(6, 8);
        A.multiply(C, tmp);
        checkAgainstRCV(tmp, vAC, rAC, cAC);

        final int[] rCtB = new int[]
        {
            6, 7, 2, 3, 4, 5, 2, 6, 1, 3, 8, 1, 3, 4, 5, 8
        };

        final int[] cCtB = new int[]
        {
            1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6
        };

        final double[] vCtB = new double[]
        {
            3, 2, -2, 3, 3, 3, 1, 2, 5, 2, 7, 1, 4, 4, 4, 1
        };

        tmp = new SparseMatrix(8, 6);
        Ct.multiply(B, tmp);
        checkAgainstRCV(tmp, vCtB, rCtB, cCtB);
        
        try
        {
            A.multiply(Ct, C);
            fail("Should have failed, matrix dimensions dont agree");
        }
        catch(final ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of multiply method, of class SparseMatrix.
     */
    @Test
    public void testMultiply_3args_2()
    {
        System.out.println("multiply");
        SparseMatrix tmp;

        final int[] rAB = new int[]
        {
            1, 3, 4, 6, 5, 2, 3, 4, 5, 1, 3, 2, 3
        };

        final int[] cAB = new int[]
        {
            1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 6, 6
        };

        final double[] vAB = new double[]
        {
            8, 6, 3, 6, -2, 9, 4, 2, 1, 2, -35, 12, -7
        };

        tmp = new SparseMatrix(6, 6);
        A.multiply(B, tmp, ex);
        checkAgainstRCV(tmp, vAB, rAB, cAB);



        final int[] rBA = new int[]
        {
            2, 5, 2, 4, 6, 4, 6, 2, 3, 4, 5, 1, 3
        };

        final int[] cBA = new int[]
        {
            1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 5, 6, 6
        };

        final double[] vBA = new double[]
        {
            3, 2, 2, 6, 1, -21, -6, 12, 3, 12, 8, 2, 5
        };

        tmp = new SparseMatrix(6, 6);
        B.multiply(A, tmp, ex);
        checkAgainstRCV(tmp, vBA, rBA, cBA);

        final int[] rAC = new int[]
        {
            3, 5, 1, 2, 2, 2, 3, 4, 1, 6, 1, 3
        };

        final int[] cAC = new int[]
        {
            1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8
        };

        final double[] vAC = new double[]
        {
            -7, 1, 1, 3, 3, 3, 2, 1, 4, 3, 1, -7
        };

        tmp = new SparseMatrix(6, 8);
        A.multiply(C, tmp, ex);
        checkAgainstRCV(tmp, vAC, rAC, cAC);

        final int[] rCtB = new int[]
        {
            6, 7, 2, 3, 4, 5, 2, 6, 1, 3, 8, 1, 3, 4, 5, 8
        };

        final int[] cCtB = new int[]
        {
            1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6
        };

        final double[] vCtB = new double[]
        {
            3, 2, -2, 3, 3, 3, 1, 2, 5, 2, 7, 1, 4, 4, 4, 1
        };

        tmp = new SparseMatrix(8, 6);
        Ct.multiply(B, tmp, ex);
        checkAgainstRCV(tmp, vCtB, rCtB, cCtB);
        
        try
        {
            A.multiply(Ct, C, ex);
            fail("Should have failed, matrix dimensions dont agree");
        }
        catch(final ArithmeticException ex)
        {
            
        }
    }
    
    @Test
    public void testMultiplyTranspose()
    {
        System.out.println("multiplyTranspose");
        final int[] rAC = new int[]
        {
            3, 5, 1, 2, 2, 2, 3, 4, 1, 6, 1, 3
        };

        final int[] cAC = new int[]
        {
            1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8
        };

        final double[] vAC = new double[]
        {
            -7, 1, 1, 3, 3, 3, 2, 1, 4, 3, 1, -7
        };

        final SparseMatrix tmp = new SparseMatrix(6, 8);
        A.multiplyTranspose(Ct, tmp);
        checkAgainstRCV(tmp, vAC, rAC, cAC);
    }
    
    @Test
    public void testMultiplyTranspose_Executor()
    {
        System.out.println("multiplyTranspose_Executor");
        final int[] rAC = new int[]
        {
            3, 5, 1, 2, 2, 2, 3, 4, 1, 6, 1, 3
        };

        final int[] cAC = new int[]
        {
            1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8
        };

        final double[] vAC = new double[]
        {
            -7, 1, 1, 3, 3, 3, 2, 1, 4, 3, 1, -7
        };

        final SparseMatrix tmp = new SparseMatrix(6, 8);
        A.multiplyTranspose(Ct, tmp, new FakeExecutor());
        checkAgainstRCV(tmp, vAC, rAC, cAC);
    }

    /**
     * Test of mutableMultiply method, of class SparseMatrix.
     */
    @Test
    public void testMutableMultiply_double()
    {
        System.out.println("mutableMultiply");
        final int[] r = new int[]
        {
            1, 3, 4, 3, 2, 1, 6, 5
        };

        final int[] c = new int[]
        {
            1, 2, 2, 3, 4, 5, 5, 6
        };

        final double[] v = new double[]
        {
            3, 6, 3, -21, 9, 12, 9, 3
        };

        A.mutableMultiply(3.0);

        checkAgainstRCV(A, v, r, c);
    }

    /**
     * Test of mutableMultiply method, of class SparseMatrix.
     */
    @Test
    public void testMutableMultiply_double_ExecutorService()
    {
        System.out.println("mutableMultiply");
        final int[] r = new int[]
        {
            1, 3, 4, 3, 2, 1, 6, 5
        };

        final int[] c = new int[]
        {
            1, 2, 2, 3, 4, 5, 5, 6
        };

        final double[] v = new double[]
        {
            3, 6, 3, -21, 9, 12, 9, 3
        };

        A.mutableMultiply(3.0, ex);

        checkAgainstRCV(A, v, r, c);
    }

    /**
     * Test of mutableTranspose method, of class SparseMatrix.
     */
    @Test
    public void testMutableTranspose()
    {
        System.out.println("mutableTranspose");
        final int[] r = new int[]
        {
            1, 5, 4, 2, 3, 2, 6, 5
        };

        final int[] c = new int[]
        {
            1, 1, 2, 3, 3, 4, 5, 6
        };

        final double[] v = new double[]
        {
            1, 4, 3, 2, -7, 1, 1, 3
        };
        
        A.mutableTranspose();
        checkAgainstRCV(A, v, r, c);
        
        try
        {
            C.mutableTranspose();
            fail("C is not square, should have failed");
        }
        catch(final Exception ex)
        {
            
        }

        
    }

    /**
     * Test of transpose method, of class SparseMatrix.
     */
    @Test
    public void testTranspose()
    {
        System.out.println("transpose");
        final int[] rAt = new int[]
        {
            1, 5, 4, 2, 3, 2, 6, 5
        };

        final int[] cAt = new int[]
        {
            1, 1, 2, 3, 3, 4, 5, 6
        };

        final double[] vAt = new double[]
        {
            1, 4, 3, 2, -7, 1, 1, 3
        };
        
        SparseMatrix tmp = new SparseMatrix(6, 6);
        A.transpose(tmp);
        checkAgainstRCV(tmp, vAt, rAt, cAt);
        
        tmp = new SparseMatrix(Ct.rows(), Ct.cols());
        C.transpose(tmp);
        
        assertTrue(tmp.equals(Ct));
    }

    /**
     * Test of transposeMultiply method, of class SparseMatrix.
     */
    @Test
    public void testTransposeMultiply_Matrix_Matrix()
    {
        System.out.println("transposeMultiply");
        SparseMatrix tmp;

        final int[] rAtB = new int[]
        {
            4, 6, 5, 2, 4, 5, 1, 2, 3, 5, 2, 3
        };

        final int[] cAtB = new int[]
        {
            1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6
        };

        final double[] vAtB = new double[]
        {
            9, 2, -6, 3, 6, 3, 2, 10, -35, 8, 6, -7
        };

        tmp = new SparseMatrix(6, 6);
        A.transposeMultiply(B, tmp);
        checkAgainstRCV(tmp, vAtB, rAtB, cAtB);



        final int[] rBtA = new int[]
        {
            5, 3, 5, 6, 5, 6, 1, 4, 2, 4, 5, 1
        };

        final int[] cBtA = new int[]
        {
            1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6
        };

        final double[] vBtA = new double[]
        {
            2, 3, 10, 6, -35, -7, 9, 6, -6, 3, 8, 2
        };

        tmp = new SparseMatrix(6, 6);
        B.transposeMultiply(A, tmp);
        checkAgainstRCV(tmp, vBtA, rBtA, cBtA);

        final int[] rAtC = new int[]
        {
            2, 3, 5, 1, 2, 5, 2, 2, 4, 6, 1, 2, 3, 5
        };

        final int[] cAtC = new int[]
        {
            1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8
        };

        final double[] vAtC = new double[]
        {
            2, -7, 3, 1, 1, 4, 1, 1, 3, 1, 1, 2, -7, 4
        };

        tmp = new SparseMatrix(6, 8);
        A.transposeMultiply(C, tmp);
        checkAgainstRCV(tmp, vAtC, rAtC, cAtC);

        final int[] rCtB = new int[]
        {
            6, 7, 2, 3, 4, 5, 2, 6, 1, 3, 8, 1, 3, 4, 5, 8
        };

        final int[] cCtB = new int[]
        {
            1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6
        };

        final double[] vCtB = new double[]
        {
            3, 2, -2, 3, 3, 3, 1, 2, 5, 2, 7, 1, 4, 4, 4, 1
        };

        tmp = new SparseMatrix(8, 6);
        C.transposeMultiply(B, tmp);
        checkAgainstRCV(tmp, vCtB, rCtB, cCtB);
        
        try
        {
            A.transposeMultiply(Ct, C);
            fail("Should have failed, matrix dimensions dont agree");
        }
        catch(final ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of transposeMultiply method, of class SparseMatrix.
     */
    @Test
    public void testTransposeMultiply_3args_1()
    {
        System.out.println("transposeMultiply");
         
        SparseMatrix tmp;

        final int[] rAtB = new int[]
        {
            4, 6, 5, 2, 4, 5, 1, 2, 3, 5, 2, 3
        };

        final int[] cAtB = new int[]
        {
            1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6
        };

        final double[] vAtB = new double[]
        {
            9, 2, -6, 3, 6, 3, 2, 10, -35, 8, 6, -7
        };

        tmp = new SparseMatrix(6, 6);
        A.transposeMultiply(B, tmp, ex);
        checkAgainstRCV(tmp, vAtB, rAtB, cAtB);



        final int[] rBtA = new int[]
        {
            5, 3, 5, 6, 5, 6, 1, 4, 2, 4, 5, 1
        };

        final int[] cBtA = new int[]
        {
            1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6
        };

        final double[] vBtA = new double[]
        {
            2, 3, 10, 6, -35, -7, 9, 6, -6, 3, 8, 2
        };

        tmp = new SparseMatrix(6, 6);
        B.transposeMultiply(A, tmp, ex);
        checkAgainstRCV(tmp, vBtA, rBtA, cBtA);

        final int[] rAtC = new int[]
        {
            2, 3, 5, 1, 2, 5, 2, 2, 4, 6, 1, 2, 3, 5
        };

        final int[] cAtC = new int[]
        {
            1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8
        };

        final double[] vAtC = new double[]
        {
            2, -7, 3, 1, 1, 4, 1, 1, 3, 1, 1, 2, -7, 4
        };

        tmp = new SparseMatrix(6, 8);
        A.transposeMultiply(C, tmp, ex);
        checkAgainstRCV(tmp, vAtC, rAtC, cAtC);

        final int[] rCtB = new int[]
        {
            6, 7, 2, 3, 4, 5, 2, 6, 1, 3, 8, 1, 3, 4, 5, 8
        };

        final int[] cCtB = new int[]
        {
            1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6
        };

        final double[] vCtB = new double[]
        {
            3, 2, -2, 3, 3, 3, 1, 2, 5, 2, 7, 1, 4, 4, 4, 1
        };

        tmp = new SparseMatrix(8, 6);
        C.transposeMultiply(B, tmp, ex);
        checkAgainstRCV(tmp, vCtB, rCtB, cCtB);
        
        try
        {
            A.transposeMultiply(Ct, C);
            fail("Should have failed, matrix dimensions dont agree");
        }
        catch(final ArithmeticException ex)
        {
            
        }
    }

    /**
     * Test of transposeMultiply method, of class SparseMatrix.
     */
    @Test
    public void testTransposeMultiply_3args_2()
    {
        System.out.println("transposeMultiply");
        final Vec b = new DenseVector(new double[]
        {
            5, 3, 3, 5, 4, 6
        });

        final Vec A2b = new DenseVector(new double[]
        {
            10, 22, -42, 18, 76, 8
        });
        final Vec B2b = new DenseVector(new double[]
        {
            34, -24, 30, 24, 50, 46
        });
        final Vec Ct2b = new DenseVector(new double[]
        {
            6, 12, 20, 10, 10, 6, 8, 16
        });
        
        DenseVector c = new DenseVector(6);
        
        c.zeroOut();
        A.transposeMultiply(2, b, c);
        assertTrue(c.equals(A2b, 1e-20));
        
        c.zeroOut();
        B.transposeMultiply(2, b, c);
        assertTrue(c.equals(B2b, 1e-20));
        
        try
        {
            c.zeroOut();
            Ct.transposeMultiply(2, b,  c);//b is wrong size
            fail("Target vector does not agre, should have failed");
        }
        catch(final Exception ex)
        {
            
        }
        
        try
        {
            c.zeroOut();
            C.transposeMultiply(2, b, c);//c is wrong size
            fail("Target vector does not agre, should have failed");
        }
        catch(final Exception ex)
        {
            
        }
        
        c = new DenseVector(8);
        c.zeroOut();
        C.transposeMultiply(2, b, c);
        assertTrue(c.equals(Ct2b, 1e-20));
    }

    /**
     * Test of getRowView method, of class SparseMatrix.
     */
    @Test
    public void testGetRowView()
    {
        System.out.println("getRowView");
        
        final Vec row = A.getRowView(0);
        row.set(0, 0.0);
        row.set(4, -1.0);
        
        assertEquals(7, A.nnz());
        assertEquals(0.0, A.get(0, 0), 1e-20);
        assertEquals(-1.0, A.get(0, 4), 1e-20);
    }

    /**
     * Test of get method, of class SparseMatrix.
     */
    @Test
    public void testGet()
    {
        System.out.println("get");
        
        assertEquals(1.0, A.get(0, 0), 1e-20);
        assertEquals(4.0, A.get(0, 4), 1e-20);
        
        assertEquals(0.0, A.get(3, 0), 1e-20);
        assertEquals(1.0, A.get(3, 1), 1e-20);
    }

    /**
     * Test of set method, of class SparseMatrix.
     */
    @Test
    public void testSet()
    {
        System.out.println("set");
        
        A.set(0, 0, 0.0);
        A.set(0, 4, -1.0);
        A.set(3, 0, -2.0);
        
        assertEquals(8, A.nnz());
        assertEquals(0.0, A.get(0, 0), 1e-20);
        assertEquals(-1.0, A.get(0, 4), 1e-20);
        
        assertEquals(-2.0, A.get(3, 0), 1e-20);
        assertEquals(1.0, A.get(3, 1), 1e-20);
    }

    /**
     * Test of increment method, of class SparseMatrix.
     */
    @Test
    public void testIncrement()
    {
        System.out.println("increment");
        
        A.increment(0, 4, -1.0);
        A.increment(3, 0, -2.0);
        
        assertEquals(9, A.nnz());
        assertEquals(1.0, A.get(0, 0), 1e-20);
        assertEquals(3.0, A.get(0, 4), 1e-20);
        
        assertEquals(-2.0, A.get(3, 0), 1e-20);
        assertEquals(1.0, A.get(3, 1), 1e-20);
    }

    /**
     * Test of rows method, of class SparseMatrix.
     */
    @Test
    public void testRows()
    {
        System.out.println("rows");
        assertEquals(6, A.rows());
        assertEquals(6, B.rows());
        assertEquals(6, C.rows());
        assertEquals(8, Ct.rows());
    }

    /**
     * Test of cols method, of class SparseMatrix.
     */
    @Test
    public void testCols()
    {
        System.out.println("cols");
        assertEquals(6, A.cols());
        assertEquals(6, B.cols());
        assertEquals(8, C.cols());
        assertEquals(6, Ct.cols());
    }

    /**
     * Test of isSparce method, of class SparseMatrix.
     */
    @Test
    public void testIsSparce()
    {
        System.out.println("isSparce");
        assertTrue(A.isSparce());
        assertTrue(B.isSparce());
        assertTrue(C.isSparce());
        assertTrue(Ct.isSparce());
    }

    /**
     * Test of swapRows method, of class SparseMatrix.
     */
    @Test
    public void testSwapRows()
    {
        System.out.println("swapRows");
        final int[] r = new int[]
        {
            6, 3, 4, 3, 2, 1, 6, 5
        };

        final int[] c = new int[]
        {
            1, 2, 2, 3, 4, 5, 5, 6
        };

        final double[] v = new double[]
        {
            1, 2, 1, -7, 3, 3, 4, 1
        };
        
        A.swapRows(0, 5);
        checkAgainstRCV(A, v, r, c);
    }

    /**
     * Test of zeroOut method, of class SparseMatrix.
     */
    @Test
    public void testZeroOut()
    {
        System.out.println("zeroOut");
        A.zeroOut();
        assertEquals(0, A.nnz());
        for(int i = 0; i < A.rows(); i++) {
          for (int j = 0; j < A.cols(); j++) {
            assertEquals(0.0, A.get(i, j), 1e-20);
          }
        }
    }

    /**
     * Test of clone method, of class SparseMatrix.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        final Matrix AClone = A.clone();
        assertTrue(AClone.equals(A));
        assertFalse(AClone == A);
        A.zeroOut();
        assertFalse(AClone.equals(A));
        assertEquals(8, AClone.nnz());
        assertEquals(0, A.nnz());
    }
    
    @Test
    public void testNnz()
    {
        System.out.println("mutableAdd");
        assertEquals(8, A.nnz());
        assertEquals(10, B.nnz());
        assertEquals(10, C.nnz());
        assertEquals(10, Ct.nnz());
    }
    
    @Test
    public void testChangeSize()
    {
        System.out.println("changeSize");
        final Matrix Acpy = A.clone();
        Acpy.changeSize(Acpy.rows()-1, Acpy.cols()-1);
        assertEquals(Acpy.rows(), A.rows()-1);
        assertEquals(Acpy.cols(), A.cols()-1);
        
        for(int i = 0; i < Acpy.rows(); i++) {
          for (int j = 0; j < Acpy.cols(); j++) {
            assertEquals(Acpy.get(i, j), A.get(i, j), 0.0);
          }
        }
        //Expand back out and make sure the values are zero on the sides
        Acpy.changeSize(Acpy.rows()+2, Acpy.cols()+2);
        assertEquals(Acpy.rows(), A.rows()+1);
        assertEquals(Acpy.cols(), A.cols()+1);
        
        for(int i = 0; i < Acpy.rows(); i++) {
          for (int j = 0; j < Acpy.cols(); j++) {
            if (i < A.rows()-1 && j < A.cols()-1) {
              assertEquals(A.get(i, j), Acpy.get(i, j), 0.0);
            } else {
              assertEquals(0.0, Acpy.get(i, j), 0.0);
            }
          }
        }
    }

    /**
     * 
     * @param tmp the matrix to check
     * @param v the value stored for each non zero index
     * @param r the row for each non zero index, 1 based
     * @param c the column for each non zero index, 1 based
     */
    private void checkAgainstRCV(final Matrix tmp, final double[] v, final int[] r, final int[] c)
    {
        assertEquals(v.length, tmp.nnz());
        for(int i = 0; i < v.length; i++) {
          assertEquals(v[i], tmp.get(r[i]-1, c[i]-1), 1e-20);
        }
    }
}
