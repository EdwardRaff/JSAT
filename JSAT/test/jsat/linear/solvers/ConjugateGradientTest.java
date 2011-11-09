/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear.solvers;

import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
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
public class ConjugateGradientTest
{
    
    public ConjugateGradientTest()
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
    }
    
    @After
    public void tearDown()
    {
    }


    @Test
    public void testSolve_4args()
    {
        System.out.println("solve");
        DenseMatrix A = new DenseMatrix(new double[][] 
        {
            {4, 1},
            {1, 3}
        });
        
        DenseVector b = DenseVector.toDenseVec(1, 2);
        
        Vec x = new DenseVector(2);
        
        x = ConjugateGradient.solve(1e-13, A, x, b);
        
        assertTrue(A.multiply(x).equals(b, 1e-10));
        
        //Test for a 5x5 matrix symmetric positive definite 
        A = new DenseMatrix(new double[][]
        {
            {1,     1,     1,     1,     1},
            {1,     8,     1,     8,     1},
            {1,     1,    27,     1,     1},
            {1,     8,     1,    64,     1},
            {1,     1,     1,     1,   125}
        });
        
        b = DenseVector.toDenseVec(1, 4, 3, 5, 2);
        
        x = new DenseVector(5);
        
        x = ConjugateGradient.solve(1e-13, A, x, b);
        
        assertTrue(A.multiply(x).equals(b, 1e-10));
    }

    @Test
    public void testSolve_Matrix_Vec()
    {
        System.out.println("solve");
        DenseMatrix A = new DenseMatrix(new double[][] 
        {
            {4, 1},
            {1, 3}
        });
        
        DenseVector b = DenseVector.toDenseVec(1, 2);
        
        Vec x = new DenseVector(2);
        
        x = ConjugateGradient.solve(A, b);
        
        assertTrue(A.multiply(x).equals(b, 1e-8));
        
        //Test for a 5x5 matrix symmetric positive definite 
        A = new DenseMatrix(new double[][]
        {
            {1,     1,     1,     1,     1},
            {1,     8,     1,     8,     1},
            {1,     1,    27,     1,     1},
            {1,     8,     1,    64,     1},
            {1,     1,     1,     1,   125}
        });
        
        b = DenseVector.toDenseVec(1, 4, 3, 5, 2);
        
        x = new DenseVector(5);
        
        x = ConjugateGradient.solve(A, b);
        
        assertTrue(A.multiply(x).equals(b, 1e-8));
    }

    @Test
    public void testSolveCGNR_4args()
    {
        System.out.println("solveCGNR");
        double eps = 1e-14;
        
        DenseMatrix A = new DenseMatrix(new double[][]
        {
            {9 ,    3,     9,     5,     9,     6,     9,     3},
            {10,    8,     3,     4,     3,     1,     1,     2},
            {5 ,    3,     2,     8,     8,     1,     6,     8},
            {1 ,    8,     3,     6,     8,     5,     5,     3},
            {1 ,    2,     6,     5,     4,     8,     0,     5},
        });
        
        DenseVector b = DenseVector.toDenseVec(1, 4, 3, 5, 2);
        Vec x = new DenseVector(A.cols());
        
        x = ConjugateGradient.solveCGNR(eps, A, x, b);
        
        assertTrue(A.multiply(x).equals(b, 1e-10));
        
        //This is under determined, exact result will not be possible
        
        A = new DenseMatrix(new double[][]
        {
            {4 ,    8,     8,     1},
            {1 ,    9,     4,     1},
            {10,    1,     9,     9},
            {0 ,    4,     2,     6},
            {8 ,    3,     3,     5},
        });
        
        x = new DenseVector(A.cols());
        x = ConjugateGradient.solveCGNR(eps, A, x, b);
        
        double error = A.multiply(x).subtract(b).pNorm(2);
        assertEquals(1.0125, error, 1e-4);//True result computed with matlab
    }

    @Test
    public void testSolveCGNR_Matrix_Vec()
    {
        System.out.println("solveCGNR");
        DenseMatrix A = new DenseMatrix(new double[][]
        {
            {9 ,    3,     9,     5,     9,     6,     9,     3},
            {10,    8,     3,     4,     3,     1,     1,     2},
            {5 ,    3,     2,     8,     8,     1,     6,     8},
            {1 ,    8,     3,     6,     8,     5,     5,     3},
            {1 ,    2,     6,     5,     4,     8,     0,     5},
        });
        
        DenseVector b = DenseVector.toDenseVec(1, 4, 3, 5, 2);
        Vec x;
        
        x = ConjugateGradient.solveCGNR(A, b);
        
        assertTrue(A.multiply(x).equals(b, 1e-8));
        
        //This is under determined, exact result will not be possible
        
        A = new DenseMatrix(new double[][]
        {
            {4 ,    8,     8,     1},
            {1 ,    9,     4,     1},
            {10,    1,     9,     9},
            {0 ,    4,     2,     6},
            {8 ,    3,     3,     5},
        });
        
        x = ConjugateGradient.solveCGNR(A, b);
        
        double error = A.multiply(x).subtract(b).pNorm(2);
        assertEquals(1.0125, error, 1e-4);//True result computed with matlab
    }

    @Test
    public void testSolve_5args()
    {
        System.out.println("solve");
        
        
        //Test for a 5x5 matrix symmetric positive definite 
        DenseMatrix A = new DenseMatrix(new double[][]
        {
            {1,     1,     1,     1,     1},
            {1,     8,     1,     8,     1},
            {1,     1,    27,     1,     1},
            {1,     8,     1,    64,     1},
            {1,     1,     1,     1,   125}
        });
        
        DenseVector b = DenseVector.toDenseVec(1, 4, 3, 5, 2);
        
        DenseMatrix Minv = new DenseMatrix(new double[][]
        {
            {1.0000,    0     ,    0     ,    0       ,  0},
            {0     ,    0.1250,    0     ,    0       ,  0},
            {0     ,    0     ,    0.0370,    0       ,  0},
            {0     ,    0     ,    0     ,    0.0156  ,  0},
            {0     ,    0     ,    0     ,    0       ,  0.0080},
        });
        
        
        
        Vec x = new DenseVector(5);
        
        x = ConjugateGradient.solve(1e-13, A, x, b, Minv);
        
        assertTrue(A.multiply(x).equals(b, 1e-3));//Our Minv only has 4 sig figs
        
        //DO it again using the identiy matrix as the precondition (which dosnt actually help, but makes sure the code is runnig right)
        
        Minv = Matrix.eye(5);
        x = ConjugateGradient.solve(1e-13, A, x, b, Minv);
        
        assertTrue(A.multiply(x).equals(b, 1e-10));
    }
}
