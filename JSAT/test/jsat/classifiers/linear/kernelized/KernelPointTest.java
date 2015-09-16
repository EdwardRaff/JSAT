package jsat.classifiers.linear.kernelized;

import static java.lang.Math.pow;
import static org.junit.Assert.assertEquals;
import java.util.List;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.distributions.kernels.KernelPoint;
import jsat.distributions.kernels.LinearKernel;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 *
 * @author Edward Raff
 */
public class KernelPointTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  List<Vec> toAdd;

  List<Vec> toTest;

  double[] coeff;

  public KernelPointTest() {
  }

  @Before
  public void setUp() {
    final Vec mean = new DenseVector(new double[] { 2.0, -1.0, 3.0 });

    final Matrix cov = new DenseMatrix(new double[][] { { 1.07142, 1.15924, 0.38842 }, { 1.15924, 1.33071, 0.51373 },
        { 0.38842, 0.51373, 0.92986 }, });

    final NormalM normal = new NormalM(mean, cov);

    final Random rand = new Random(42);
    toAdd = normal.sample(10, rand);
    toTest = normal.sample(10, rand);
    coeff = new double[toAdd.size()];
    for (int i = 0; i < coeff.length; i++) {
      coeff[i] = Math.round(rand.nextDouble() * 9 + 0.5);
    }
    for (int i = 0; i < coeff.length; i++) {
      if (i % 2 != 0) {
        coeff[i] *= -1;
      }
    }
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testDistance_KernelPoint() {
    System.out.println("distance_KernelPoint");
    final KernelPoint kpSimpleA = new KernelPoint(new LinearKernel(0), 1e-2);
    final KernelPoint kpCoeffA = new KernelPoint(new LinearKernel(0), 1e-2);

    final KernelPoint kpSimpleB = new KernelPoint(new LinearKernel(0), 1e-2);
    final KernelPoint kpCoeffB = new KernelPoint(new LinearKernel(0), 1e-2);

    final EuclideanDistance d = new EuclideanDistance();

    for (int i = 0; i < toAdd.size(); i++) {
      final Vec sumSimpleA = toAdd.get(0).clone();
      final Vec sumCoeffA = toAdd.get(0).multiply(coeff[0]);
      for (int ii = 1; ii < i + 1; ii++) {
        sumSimpleA.mutableAdd(toAdd.get(ii));
        sumCoeffA.mutableAdd(coeff[ii], toAdd.get(ii));
      }

      final Vec sumSimpleB = toTest.get(0).clone();
      final Vec sumCoeffB = toTest.get(0).multiply(coeff[0]);
      for (int ii = 1; ii < i + 1; ii++) {
        sumSimpleB.mutableAdd(toTest.get(ii));
        sumCoeffB.mutableAdd(coeff[ii], toTest.get(ii));
      }

      kpSimpleA.mutableAdd(toAdd.get(i));
      kpCoeffA.mutableAdd(coeff[i], toAdd.get(i));

      kpSimpleB.mutableAdd(toTest.get(i));
      kpCoeffB.mutableAdd(coeff[i], toTest.get(i));

      assertEquals(0.0, kpSimpleA.dist(kpSimpleA), 1e-2 * 4);
      assertEquals(0.0, kpSimpleB.dist(kpSimpleB), 1e-2 * 4);
      assertEquals(0.0, kpCoeffA.dist(kpCoeffA), 1e-2 * 4);
      assertEquals(0.0, kpCoeffB.dist(kpCoeffB), 1e-2 * 4);

      assertEquals(d.dist(sumSimpleA, sumSimpleB), kpSimpleA.dist(kpSimpleB), 1e-2 * 4);
      assertEquals(d.dist(sumSimpleA, sumCoeffA), kpSimpleA.dist(kpCoeffA), 1e-2 * 4);
      assertEquals(d.dist(sumSimpleA, sumCoeffB), kpSimpleA.dist(kpCoeffB), 1e-2 * 4);

      assertEquals(d.dist(sumCoeffA, sumSimpleB), kpCoeffA.dist(kpSimpleB), 1e-2 * 4);
      assertEquals(d.dist(sumCoeffB, sumSimpleB), kpCoeffB.dist(kpSimpleB), 1e-2 * 4);

      final KernelPoint kpSimpleAClone = kpSimpleA.clone();
      final KernelPoint kpSimpleBClone = kpSimpleB.clone();
      kpSimpleAClone.mutableMultiply(1.0 / (i + 1));
      kpSimpleBClone.mutableMultiply(1.0 / (i + 1));

      assertEquals(d.dist(sumSimpleA.divide(i + 1), sumSimpleB.divide(i + 1)), kpSimpleAClone.dist(kpSimpleBClone),
          1e-2 * 4);
      assertEquals(d.dist(sumSimpleA.divide(i + 1), sumCoeffA), kpSimpleAClone.dist(kpCoeffA), 1e-2 * 4);
      assertEquals(d.dist(sumSimpleA.divide(i + 1), sumCoeffB), kpSimpleAClone.dist(kpCoeffB), 1e-2 * 4);

      assertEquals(d.dist(sumCoeffA, sumSimpleB.divide(i + 1)), kpCoeffA.dist(kpSimpleBClone), 1e-2 * 4);
      assertEquals(d.dist(sumCoeffB, sumSimpleB.divide(i + 1)), kpCoeffB.dist(kpSimpleBClone), 1e-2 * 4);
    }
  }

  /**
   * Test of dist method, of class KernelPoint.
   */
  @Test
  public void testDistance_Vec() {
    System.out.println("distance_Vec");
    final KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
    final KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);

    final EuclideanDistance d = new EuclideanDistance();

    for (int i = 0; i < toAdd.size(); i++) {
      final Vec sumSimple = toAdd.get(0).clone();
      final Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
      for (int ii = 1; ii < i + 1; ii++) {
        sumSimple.mutableAdd(toAdd.get(ii));
        sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
      }
      kpSimple.mutableAdd(toAdd.get(i));
      kpCoeff.mutableAdd(coeff[i], toAdd.get(i));

      for (final Vec v : toTest) {
        final double expectedSimple = d.dist(sumSimple, v);
        final double expectedCoeff = d.dist(sumCoeff, v);

        assertEquals(expectedSimple, kpSimple.dist(v), 1e-2 * 4);
        assertEquals(expectedCoeff, kpCoeff.dist(v), 1e-2 * 4);

        final KernelPoint kp0 = kpSimple.clone();
        final KernelPoint kp1 = kpCoeff.clone();

        for (int j = i + 1; j < coeff.length; j++) {
          kp0.mutableAdd(toAdd.get(j));
          kp1.mutableAdd(coeff[j], toAdd.get(j));
        }

        for (int j = i + 1; j < coeff.length; j++) {
          kp0.mutableAdd(-1, toAdd.get(j));
          kp1.mutableAdd(-coeff[j], toAdd.get(j));
        }

        assertEquals(expectedSimple, kp0.dist(v), 1e-2 * 4);
        assertEquals(expectedCoeff, kp1.dist(v), 1e-2 * 4);

        kp0.mutableMultiply(1.0 / (i + 1));
        kp1.mutableMultiply(1.0 / (i + 1));

        assertEquals(d.dist(sumSimple.divide(i + 1), v), kp0.dist(v), 1e-2 * 4);
        assertEquals(d.dist(sumCoeff.divide(i + 1), v), kp1.dist(v), 1e-2 * 4);
      }
    }
  }

  @Test
  public void testDot_KernelPoint() {
    System.out.println("dot_KernelPoint");
    final KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
    final KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);

    for (int i = 0; i < toAdd.size(); i++) {
      final Vec sumSimple = toAdd.get(0).clone();
      final Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
      for (int ii = 1; ii < i + 1; ii++) {
        sumSimple.mutableAdd(toAdd.get(ii));
        sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
      }
      kpSimple.mutableAdd(toAdd.get(i));
      kpCoeff.mutableAdd(coeff[i], toAdd.get(i));

      final double expectedSimple = sumSimple.dot(sumSimple);
      final double expectedCoeff = sumCoeff.dot(sumCoeff);
      final double expectedSC = sumSimple.dot(sumCoeff);

      assertEquals(expectedSimple, kpSimple.dot(kpSimple), 1e-2 * 4);
      assertEquals(expectedCoeff, kpCoeff.dot(kpCoeff), 1e-2 * 4);
      assertEquals(expectedSC, kpSimple.dot(kpCoeff), 1e-2 * 4);

      final KernelPoint kp0 = kpSimple.clone();
      final KernelPoint kp1 = kpCoeff.clone();

      for (int j = i + 1; j < coeff.length; j++) {
        kp0.mutableAdd(toAdd.get(j));
        kp1.mutableAdd(coeff[j], toAdd.get(j));
      }

      for (int j = i + 1; j < coeff.length; j++) {
        kp0.mutableAdd(-1, toAdd.get(j));
        kp1.mutableAdd(-coeff[j], toAdd.get(j));
      }

      assertEquals(expectedSimple, kp0.dot(kpSimple), 1e-2 * 4);
      assertEquals(expectedCoeff, kp1.dot(kpCoeff), 1e-2 * 4);

      assertEquals(expectedSC, kp0.dot(kp1), 1e-2 * 4);
      assertEquals(expectedSC, kp1.dot(kp0), 1e-2 * 4);
      assertEquals(expectedSC, kp0.dot(kpCoeff), 1e-2 * 4);
      assertEquals(expectedSC, kpSimple.dot(kp1), 1e-2 * 4);

      kp0.mutableMultiply(1.0 / (i + 1));
      kp1.mutableMultiply(1.0 / (i + 1));

      assertEquals(expectedSimple / (i + 1), kp0.dot(kpSimple), 1e-2 * 4);
      assertEquals(expectedCoeff / (i + 1), kp1.dot(kpCoeff), 1e-2 * 4);

      assertEquals(expectedSC / pow(i + 1, 2), kp0.dot(kp1), 1e-2 * 4);
      assertEquals(expectedSC / pow(i + 1, 2), kp1.dot(kp0), 1e-2 * 4);
      assertEquals(expectedSC / (i + 1), kp0.dot(kpCoeff), 1e-2 * 4);
      assertEquals(expectedSC / (i + 1), kpSimple.dot(kp1), 1e-2 * 4);
    }

  }

  /**
   * Test of dot method, of class KernelPoint.
   */
  @Test
  public void testDot_Vec() {
    System.out.println("dot_Vec");
    final KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
    final KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);

    for (int i = 0; i < toAdd.size(); i++) {
      final Vec sumSimple = toAdd.get(0).clone();
      final Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
      for (int ii = 1; ii < i + 1; ii++) {
        sumSimple.mutableAdd(toAdd.get(ii));
        sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
      }
      kpSimple.mutableAdd(toAdd.get(i));
      kpCoeff.mutableAdd(coeff[i], toAdd.get(i));

      for (final Vec v : toTest) {
        final double expectedSimple = sumSimple.dot(v);
        final double expectedCoeff = sumCoeff.dot(v);

        assertEquals(expectedSimple, kpSimple.dot(v), 1e-2 * 4);
        assertEquals(expectedCoeff, kpCoeff.dot(v), 1e-2 * 4);

        final KernelPoint kp0 = kpSimple.clone();
        final KernelPoint kp1 = kpCoeff.clone();

        for (int j = i + 1; j < coeff.length; j++) {
          kp0.mutableAdd(toAdd.get(j));
          kp1.mutableAdd(coeff[j], toAdd.get(j));
        }

        for (int j = i + 1; j < coeff.length; j++) {
          kp0.mutableAdd(-1, toAdd.get(j));
          kp1.mutableAdd(-coeff[j], toAdd.get(j));
        }

        assertEquals(expectedSimple, kp0.dot(v), 1e-2 * 4);
        assertEquals(expectedCoeff, kp1.dot(v), 1e-2 * 4);

        kp0.mutableMultiply(1.0 / (i + 1));
        kp1.mutableMultiply(1.0 / (i + 1));

        assertEquals(expectedSimple / (i + 1), kp0.dot(v), 1e-2 * 4);
        assertEquals(expectedCoeff / (i + 1), kp1.dot(v), 1e-2 * 4);
      }
    }
  }

  /**
   * Test of getSqrdNorm method, of class KernelPoint.
   */
  @Test
  public void testGetSqrdNorm() {
    System.out.println("getSqrdNorm");
    final KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
    final KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);

    for (int i = 0; i < toAdd.size(); i++) {
      final Vec sumSimple = toAdd.get(0).clone();
      final Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
      for (int ii = 1; ii < i + 1; ii++) {
        sumSimple.mutableAdd(toAdd.get(ii));
        sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
      }
      kpSimple.mutableAdd(toAdd.get(i));
      kpCoeff.mutableAdd(coeff[i], toAdd.get(i));

      final double expectedSimple = Math.pow(sumSimple.pNorm(2), 2);
      final double expectedCoeff = Math.pow(sumCoeff.pNorm(2), 2);

      assertEquals(expectedSimple, kpSimple.getSqrdNorm(), 1e-2 * 4);
      assertEquals(expectedCoeff, kpCoeff.getSqrdNorm(), 1e-2 * 4);

      final KernelPoint kp0 = kpSimple.clone();
      final KernelPoint kp1 = kpCoeff.clone();

      for (int j = i + 1; j < coeff.length; j++) {
        kp0.mutableAdd(toAdd.get(j));
        kp1.mutableAdd(coeff[j], toAdd.get(j));
      }

      for (int j = i + 1; j < coeff.length; j++) {
        kp0.mutableAdd(-1, toAdd.get(j));
        kp1.mutableAdd(-coeff[j], toAdd.get(j));
      }

      assertEquals(expectedSimple, kp0.getSqrdNorm(), 1e-2 * 4);
      assertEquals(expectedCoeff, kp1.getSqrdNorm(), 1e-2 * 4);

      kp0.mutableMultiply(1.0 / (i + 1));
      kp1.mutableMultiply(1.0 / (i + 1));

      assertEquals(expectedSimple / pow(i + 1, 2), kp0.getSqrdNorm(), 1e-2 * 4);
      assertEquals(expectedCoeff / pow(i + 1, 2), kp1.getSqrdNorm(), 1e-2 * 4);
    }

  }
}
