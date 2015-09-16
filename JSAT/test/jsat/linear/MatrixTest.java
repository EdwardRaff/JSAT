/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.math.OnLineStatistics;
import jsat.utils.SystemInfo;
import jsat.utils.random.XOR128;

/**
 *
 * @author Edward Raff
 */
public class MatrixTest {

  private static ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public MatrixTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of diag method, of class Matrix.
   */
  @Test
  public void testDiag() {
    System.out.println("diag");

    assertEquals(Matrix.eye(5), Matrix.diag(new ConstantVector(1.0, 5)));

  }

  /**
   * Test of diagMult method, of class Matrix.
   */
  @Test
  public void testDiagMult_Matrix_Vec() {
    // TODO add diagonal test case
    System.out.println("diagMult");
    final Matrix A = new DenseMatrix(new double[][] { { 0, 8, 7, 5, 5 }, { 6, 10, 4, 8, 4 }, { 10, 7, 5, 8, 6 },
        { 5, 2, 2, 5, 5 }, { 6, 7, 10, 5, 8 }, });

    final Vec b = new DenseVector(new double[] { 4, -3, 3, -4, 2 });

    final double[][] expected = new double[][] { { 0, -24, 21, -20, 10 }, { 24, -30, 12, -32, 8 },
        { 40, -21, 15, -32, 12 }, { 20, -6, 6, -20, 10 }, { 24, -21, 30, -20, 16 }, };
    Matrix.diagMult(A, b);
    assertEquals(new DenseMatrix(expected), A);
  }

  /**
   * Test of diagMult method, of class Matrix.
   */
  @Test
  public void testDiagMult_Vec_Matrix() {
    // TODO add diagonal test case
    System.out.println("diagMult");
    final Matrix A = new DenseMatrix(new double[][] { { 0, 8, 7, 5, 5 }, { 6, 10, 4, 8, 4 }, { 10, 7, 5, 8, 6 },
        { 5, 2, 2, 5, 5 }, { 6, 7, 10, 5, 8 }, });

    final Vec b = new DenseVector(new double[] { 4, -3, 3, -4, 2 });

    final double[][] expected = new double[][] { { 0, 32, 28, 20, 20 }, { -18, -30, -12, -24, -12 },
        { 30, 21, 15, 24, 18 }, { -20, -8, -8, -20, -20 }, { 12, 14, 20, 10, 16 }, };

    Matrix.diagMult(b, A);
    assertEquals(new DenseMatrix(expected), A);
  }

  /**
   * Test of eye method, of class Matrix.
   */
  @Test
  public void testEye() {
    System.out.println("eye");

    for (int k = 1; k < 10; k++) {
      final Matrix I = Matrix.eye(k);
      for (int i = 0; i < I.rows(); i++) {
        for (int j = 0; j < I.cols(); j++) {
          if (i == j) {
            assertEquals(1.0, I.get(i, j), 0.0);
          } else {
            assertEquals(0.0, I.get(i, j), 0.0);
          }
        }
      }
    }
  }

  /**
   * Test of isSymmetric method, of class Matrix.
   */
  @Test
  public void testIsSymmetric_Matrix() {
    System.out.println("isSymmetric");
    final Matrix A = Matrix.eye(5);

    assertTrue(Matrix.isSymmetric(A));

    A.set(3, 4, 2.0);
    A.set(4, 3, 2.0);

    assertTrue(Matrix.isSymmetric(A));

    A.set(3, 2, 2);

    assertFalse(Matrix.isSymmetric(A));

  }

  /**
   * Test of isSymmetric method, of class Matrix.
   */
  @Test
  public void testIsSymmetric_Matrix_double() {
    System.out.println("isSymmetric");
    final Matrix A = Matrix.eye(5);

    assertTrue(Matrix.isSymmetric(A, 0.0));

    A.set(3, 4, 2.0);
    A.set(4, 3, 2.0);

    assertTrue(Matrix.isSymmetric(A, 0.0));

    A.set(3, 2, 0.01);

    assertFalse(Matrix.isSymmetric(A, 0.0));
    assertTrue(Matrix.isSymmetric(A, 0.1));
  }

  /**
   * Test of OuterProductUpdate method, of class Matrix.
   */
  @Test
  public void testOuterProductUpdate_4args() {
    System.out.println("OuterProductUpdate");
    final Matrix A = Matrix.eye(4);
    final Vec x = new DenseVector(new double[] { 1, 2, 3, 4 });
    final Vec y = new DenseVector(new double[] { 5, 6, 7, 8 });
    final double c = 2.0;

    final double[][] expected = new double[][] { { 11, 12, 14, 16 }, { 20, 25, 28, 32 }, { 30, 36, 43, 48 },
        { 40, 48, 56, 65 }, };

    Matrix.OuterProductUpdate(A, x, y, c);

    for (int i = 0; i < expected.length; i++) {
      for (int j = 0; j < expected.length; j++) {
        assertEquals(expected[i][j], A.get(i, j), 0.0);
      }
    }
  }

  /**
   * Test of OuterProductUpdate method, of class Matrix.
   */
  @Test
  public void testOuterProductUpdate_5args() {
    System.out.println("OuterProductUpdate");
    final Matrix A = Matrix.eye(4);
    final Vec x = new DenseVector(new double[] { 1, 2, 3, 4 });
    final Vec y = new DenseVector(new double[] { 5, 6, 7, 8 });
    final double c = 2.0;

    final double[][] expected = new double[][] { { 11, 12, 14, 16 }, { 20, 25, 28, 32 }, { 30, 36, 43, 48 },
        { 40, 48, 56, 65 }, };
    Matrix.OuterProductUpdate(A, x, y, c, ex);

    for (int i = 0; i < expected.length; i++) {
      for (int j = 0; j < expected.length; j++) {
        assertEquals(expected[i][j], A.get(i, j), 0.0);
      }
    }
  }

  /**
   * Test of pascal method, of class Matrix.
   */
  @Test
  public void testPascal() {
    System.out.println("pascal");

    final Matrix P = Matrix.pascal(6);

    for (int i = 0; i < P.rows(); i++) {
      assertEquals(1.0, P.get(i, 0), 0.0);
      assertEquals(1.0, P.get(0, i), 0.0);
    }

    for (int i = 1; i < P.rows(); i++) {
      for (int j = 1; j < P.cols(); j++) {
        assertEquals(P.get(i - 1, j) + P.get(i, j - 1), P.get(i, j), 0.0);
      }
    }

  }

  /**
   * Test of random method, of class Matrix.
   */
  @Test
  public void testRandom() {
    System.out.println("random");
    final int rows = 100;
    final int cols = 100;
    final Random rand = new XOR128();

    final DenseMatrix result = Matrix.random(rows, cols, rand);
    final OnLineStatistics stats = new OnLineStatistics();
    for (int i = 0; i < result.rows(); i++) {
      for (int j = 0; j < result.cols(); j++) {
        stats.add(result.get(i, j));
      }
    }
    // if its all random from [0, 1], the mean should be 0.5
    assertEquals(0.5, stats.getMean(), 0.05);
  }

}
