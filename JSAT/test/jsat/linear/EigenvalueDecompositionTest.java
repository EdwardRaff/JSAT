/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class EigenvalueDecompositionTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  /**
   * A 5x5 matrix whos eigen values are complex
   */
  Matrix A;

  /**
   * A 3x3 matrix whos eigen values are real
   */
  Matrix B;

  /**
   * A symmetric 5x5 matrix
   */
  Matrix pascal5;

  public EigenvalueDecompositionTest() {
  }

  /**
   * Checks if the property of the eigen decomposition [V, D] = eig(A) that A*V
   * = V*D holds.
   *
   * @param evd
   *          the eigen value decomposition
   * @param A
   *          the original matrix
   * @param eps
   *          the allowable difference in results
   * @return <t>true</tt> if the decomposition was correct
   */
  private boolean eigenResultsRight(final EigenValueDecomposition evd, final Matrix A, final double eps) {
    return A.multiply(evd.getV()).equals(evd.getV().multiply(evd.getD()), eps);
  }

  @Before
  public void setUp() {
    A = new DenseMatrix(new double[][] { { 1, 5, 4, 8, 9 }, { 1, 5, 7, 3, 7 }, { 0, 3, 8, 5, 6 }, { 3, 8, 0, 7, 0 },
        { 1, 9, 2, 9, 6 } });

    B = new DenseMatrix(new double[][] { { -149, -50, -154 }, { 537, 180, 546 }, { -27, -9, -25 }, });

    pascal5 = Matrix.pascal(5);
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testImageEigen5() {
    System.out.println("testImageEigen5");
    final EigenValueDecomposition evd = new EigenValueDecomposition(A);
    assertTrue(evd.isComplex());
  }

  @Test
  public void testNonSquare() {
    System.out.println("testNonSquare");
    EigenValueDecomposition evd = null;
    try {
      evd = new EigenValueDecomposition(new DenseMatrix(10, 12));
      fail("Can not take the EVD of a non square matrix");
    } catch (final Exception e) {
    }
  }

  @Test
  public void testPascal5() {
    System.out.println("testPascal5");
    final EigenValueDecomposition evd = new EigenValueDecomposition(pascal5);
    assertFalse(evd.isComplex());

    assertTrue(eigenResultsRight(evd, pascal5, 1e-8));
  }

  @Test
  public void testRealEigen3() {
    System.out.println("testRealEigen3");
    final EigenValueDecomposition evd = new EigenValueDecomposition(B);
    assertFalse(evd.isComplex());

    assertTrue(eigenResultsRight(evd, B, 1e-8));
  }

  @Test
  public void testRealRandomGenerated() {
    System.out.println("testRealRandomGenerated");
    // Generate a eigen value decomposition backwards, randomly creating V and D
    // and then working to A, then decompositing A and checking the results
    final Random rand = new Random(123);

    final Matrix V = Matrix.random(7, 7, rand);
    final Matrix D = Matrix.diag(Vec.random(7, rand).multiply(10));
    final SingularValueDecomposition svd = new SingularValueDecomposition(V.clone());
    final Matrix A = V.multiply(D).multiply(svd.getPseudoInverse());
    final EigenValueDecomposition evd = new EigenValueDecomposition(A);
    assertTrue(eigenResultsRight(evd, A, 1e-6));
  }

  @Test
  public void testRealRandomGeneratedLarge() {
    System.out.println("testRealRandomGeneratedLarge");
    // Generate a eigen value decomposition backwards, randomly creating V and D
    // and then working to A, then decompositing A and checking the results
    final Random rand = new Random(123);

    final Matrix V = Matrix.random(70, 70, rand);
    final Matrix D = Matrix.diag(Vec.random(70, rand).multiply(10));
    final SingularValueDecomposition svd = new SingularValueDecomposition(V.clone());
    final Matrix A = V.multiply(D).multiply(svd.getPseudoInverse());
    final EigenValueDecomposition evd = new EigenValueDecomposition(A);
    assertTrue(eigenResultsRight(evd, A, 1e-6));
  }

  @Test
  public void testSymmetric70() {
    // Cant use pascal for large b/c it becomes unstable
    System.out.println("testSymmetric70");
    final Random rand = new Random(123);
    final Matrix SYM = new DenseMatrix(70, 70);
    for (int i = 0; i < SYM.rows(); i++) {
      SYM.set(i, i, rand.nextDouble() * 10);
      for (int j = i + 1; j < SYM.cols(); j++) {
        SYM.set(i, j, rand.nextDouble() * 10);
        SYM.set(j, i, SYM.get(i, j));
      }
    }
    final EigenValueDecomposition evd = new EigenValueDecomposition(SYM);
    assertFalse(evd.isComplex());

    assertTrue(eigenResultsRight(evd, SYM, 1e-8));
  }

}
