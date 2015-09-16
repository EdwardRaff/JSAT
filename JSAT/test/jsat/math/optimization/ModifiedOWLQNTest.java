package jsat.math.optimization;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import java.util.concurrent.ExecutorService;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.linear.LinearTools;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.SoftmaxLoss;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.math.FunctionVec;

/**
 *
 * @author Edward Raff
 */
public class ModifiedOWLQNTest {

  public class GradFunction implements FunctionVec {

    private final ClassificationDataSet D;
    private final LossFunc loss;

    public GradFunction(final ClassificationDataSet D, final LossFunc loss) {
      this.D = D;
      this.loss = loss;
    }

    @Override
    public Vec f(final double... x) {
      return f(DenseVector.toDenseVec(x));
    }

    @Override
    public Vec f(final Vec w) {
      final Vec s = w.clone();
      f(w, s);
      return s;
    }

    @Override
    public Vec f(final Vec w, Vec s) {
      if (s == null) {
        s = w.clone();
      }
      s.zeroOut();
      double weightSum = 0;
      for (int i = 0; i < D.getSampleSize(); i++) {
        final DataPoint dp = D.getDataPoint(i);
        final Vec x = dp.getNumericalValues();
        final double y = D.getDataPointCategory(i) * 2 - 1;
        s.mutableAdd(loss.getDeriv(w.dot(x), y) * dp.getWeight(), x);
        weightSum += dp.getWeight();
      }
      s.mutableDivide(weightSum);
      return s;
    }

    @Override
    public Vec f(final Vec x, final Vec s, final ExecutorService ex) {
      return f(x, s);
    }
  }

  public class LossFunction extends FunctionBase {

    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private final ClassificationDataSet D;
    private final LossFunc loss;

    public LossFunction(final ClassificationDataSet D, final LossFunc loss) {
      this.D = D;
      this.loss = loss;
    }

    @Override
    public double f(final Vec w) {
      double sum = 0;
      double weightSum = 0;
      for (int i = 0; i < D.getSampleSize(); i++) {
        final DataPoint dp = D.getDataPoint(i);
        final Vec x = dp.getNumericalValues();
        final double y = D.getDataPointCategory(i) * 2 - 1;
        sum += loss.getLoss(w.dot(x), y) * dp.getWeight();
        weightSum += dp.getWeight();
      }

      return sum / weightSum;
    }
  }

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public ModifiedOWLQNTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of optimize method, of class mOWLQN.
   */
  @Test
  public void testOptimize() {
    System.out.println("optimize");
    final Random rand = new Random();
    final Vec x0 = new DenseVector(10);
    for (int i = 0; i < x0.length(); i++) {
      x0.set(i, rand.nextDouble() + 0.5);
    }

    final RosenbrockFunction f = new RosenbrockFunction();
    final FunctionVec fp = f.getDerivative();
    final ModifiedOWLQN instance = new ModifiedOWLQN();
    instance.setLambda(0.0);
    instance.setMaximumIterations(500);

    final Vec w = new DenseVector(x0.length());
    instance.optimize(1e-8, w, x0, f, fp, null);

    for (int i = 0; i < w.length(); i++) {
      assertEquals(1.0, w.get(i), 1e-2);
    }
    assertEquals(0.0, f.f(w), 1e-3);
  }

  @Test
  public void testOptimizeGDsteps() {
    System.out.println("optimize");
    final Random rand = new Random();
    final Vec x0 = new DenseVector(10);
    for (int i = 0; i < x0.length(); i++) {
      x0.set(i, rand.nextDouble() + 0.5);
    }

    final RosenbrockFunction f = new RosenbrockFunction();
    final FunctionVec fp = f.getDerivative();
    final ModifiedOWLQN instance = new ModifiedOWLQN();
    instance.setLambda(0.0);
    instance.setEps(Double.MAX_VALUE);
    instance.setMaximumIterations(2000);

    final Vec w = new DenseVector(x0.length());
    instance.optimize(1e-4, w, x0, f, fp, null);

    for (int i = 0; i < w.length(); i++) {
      assertEquals(1.0, w.get(i), 0.5);
    }
    assertEquals(0.0, f.f(w), 1e-1);
  }

  @Test
  public void testOptimizeReg() {
    System.out.println("optimize");
    final Random rand = new Random();

    final ClassificationDataSet data = new ClassificationDataSet(6, new CategoricalData[0], new CategoricalData(2));

    for (int i = 0; i < 1000; i++) {
      final double Z1 = rand.nextDouble() * 20 - 10;
      final double Z2 = rand.nextDouble() * 20 - 10;
      final Vec v = DenseVector.toDenseVec(Z1 + rand.nextGaussian() / 10, -Z1 + rand.nextGaussian() / 10,
          Z1 + rand.nextGaussian() / 10, -Z2 + rand.nextGaussian() / 10, Z2 + rand.nextGaussian() / 10,
          -Z2 + rand.nextGaussian() / 10);
      data.addDataPoint(v, (int) (Math.signum(Z1 + 0.1 * Z2) + 1) / 2);
    }

    final Vec x0 = new DenseVector(data.getNumNumericalVars());
    for (int i = 0; i < x0.length(); i++) {
      x0.set(i, rand.nextDouble() * 2 - 1);
    }

    final double lambda = LinearTools.maxLambdaLogisticL1(data);

    final Function f = new LossFunction(data, new SoftmaxLoss());
    final FunctionVec fp = new GradFunction(data, new SoftmaxLoss());
    ModifiedOWLQN instance = new ModifiedOWLQN();

    instance.setLambda(lambda / 2);
    instance.setMaximumIterations(500);

    final Vec w = new DenseVector(x0.length());
    instance.optimize(1e-4, w, x0, f, fp, null);

    assertTrue(w.nnz() <= 3);
    for (final IndexValue iv : w) {
      assertTrue(iv.getIndex() < 3);
      if (iv.getIndex() % 2 == 0) {
        assertTrue(iv.getValue() > 0);
      } else {
        assertTrue(iv.getValue() < 0);
      }
    }

    // do it again, but don't regularize one value
    instance.setLambdaMultipler(DenseVector.toDenseVec(1, 1, 1, 1, 0, 1));
    x0.zeroOut();
    instance = instance.clone();
    instance.optimize(1e-4, w, x0, f, fp, null);

    assertTrue(w.nnz() <= 4);
    for (final IndexValue iv : w) {
      if (iv.getIndex() % 2 == 0) {
        assertTrue(iv.getValue() > 0);
      } else {
        assertTrue(iv.getValue() < 0);
      }
    }
    assertEquals(0.0, w.get(5), 0.0);
    assertEquals(0.0, w.get(3), 0.0);
    assertTrue(w.get(4) > 0);

  }
}
