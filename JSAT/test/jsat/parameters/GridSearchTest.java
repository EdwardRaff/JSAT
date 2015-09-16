/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.parameters;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.WarmClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.linear.DenseVector;
import jsat.parameters.Parameter.WarmParameter;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.regression.WarmRegressor;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class GridSearchTest {

  /**
   * This model is dumb. It always returns the same thing, unless the parameters
   * are set to specific values. Then it returns a 2nd option as well.
   */
  static class DumbModel implements WarmClassifier, WarmRegressor, Parameterized {

    /**
     *
     */
    private static final long serialVersionUID = 1L;
    int param1;
    double param2;
    int param3;
    boolean wasWarmStarted = false;

    @Override
    public CategoricalResults classify(final DataPoint data) {
      // range check done so that RandomSearch can re-use this class for its own
      // test
      final boolean param2InRange = 1.5 < param2 && param2 < 2.5;
      if (param1 == 1 && param2InRange && param3 == 3) {
        if (data.getNumericalValues().get(0) < 0) {
          return new CategoricalResults(new double[] { 0.0, 1.0 });
        }
      }
      return new CategoricalResults(new double[] { 1.0, 0.0 });
    }

    @Override
    public DumbModel clone() {
      final DumbModel toRet = new DumbModel();
      toRet.param1 = param1;
      toRet.param2 = param2;
      toRet.param3 = param3;
      toRet.wasWarmStarted = wasWarmStarted;
      return toRet;
    }

    public int getParam1() {
      return param1;
    }

    public double getParam2() {
      return param2;
    }

    public int getParam3() {
      return param3;
    }

    @Override
    public Parameter getParameter(final String paramName) {
      return Parameter.toParameterMap(getParameters()).get(paramName);
    }

    @Override
    public List<Parameter> getParameters() {
      return Parameter.getParamsFromMethods(this);
    }

    public Distribution guessParam1(final DataSet d) {
      return new UniformDiscrete(0, 5);
    }

    public Distribution guessParam2(final DataSet d) {
      return new Uniform(0.0, 5.0);
    }

    public Distribution guessParam3(final DataSet d) {
      return new UniformDiscrete(0, 5);
    }

    @Override
    public double regress(final DataPoint data) {
      // range check done so that RandomSearch can re-use this class for its own
      // test
      final boolean param2InRange = 1.5 < param2 && param2 < 2.5;
      if (param1 == 1 && param2InRange && param3 == 3) {
        if (data.getNumericalValues().get(0) < 0) {
          return 1;
        }
      }
      return 0;
    }

    public void setParam1(final int param1) {
      this.param1 = param1;
    }

    public void setParam2(final double param2) {
      this.param2 = param2;
    }

    @WarmParameter(prefLowToHigh = false)
    public void setParam3(final int param3) {
      this.param3 = param3;
    }

    @Override
    public boolean supportsWeightedData() {
      return true;
    }

    @Override
    public void train(final RegressionDataSet dataSet) {
      wasWarmStarted = false;
    }

    @Override
    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
      wasWarmStarted = false;
    }

    @Override
    public void train(final RegressionDataSet dataSet, final Regressor warmSolution) {
      wasWarmStarted = ((DumbModel) warmSolution).param3 == param3;
    }

    @Override
    public void train(final RegressionDataSet dataSet, final Regressor warmSolution, final ExecutorService threadPool) {
      wasWarmStarted = ((DumbModel) warmSolution).param3 == param3;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet) {
      wasWarmStarted = false;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final Classifier warmSolution) {
      wasWarmStarted = ((DumbModel) warmSolution).param3 == param3;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final Classifier warmSolution,
        final ExecutorService threadPool) {
      wasWarmStarted = ((DumbModel) warmSolution).param3 == param3;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
      wasWarmStarted = false;
    }

    @Override
    public boolean warmFromSameDataOnly() {
      return false;
    }

  }

  static ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  ClassificationDataSet classData;

  RegressionDataSet regData;

  public GridSearchTest() {
  }

  @Before
  public void setUp() {
    classData = new ClassificationDataSet(1, new CategoricalData[0], new CategoricalData(2));
    for (int i = 0; i < 100; i++) {
      classData.addDataPoint(DenseVector.toDenseVec(1.0 * i), 0);
    }
    for (int i = 0; i < 100; i++) {
      classData.addDataPoint(DenseVector.toDenseVec(-1.0 * i), 1);
    }

    regData = new RegressionDataSet(1, new CategoricalData[0]);
    for (int i = 0; i < 100; i++) {
      regData.addDataPoint(DenseVector.toDenseVec(1.0 * i), 0);
    }
    for (int i = 0; i < 100; i++) {
      regData.addDataPoint(DenseVector.toDenseVec(-1.0 * i), 1);
    }

  }

  @After
  public void tearDown() {
  }

  @Test
  public void testClassification() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Classifier) new DumbModel(), 5);
    instance.setUseWarmStarts(false);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.trainC(classData);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedClassifier();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertFalse(model.wasWarmStarted);
  }

  @Test
  public void testClassificationAutoAdd() {
    System.out.println("classificationAutoAdd");
    GridSearch instance = new GridSearch((Classifier) new DumbModel(), 5);
    instance.setUseWarmStarts(false);

    instance.autoAddParameters(classData);

    instance = instance.clone();
    instance.trainC(classData);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedClassifier();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.5);
    assertEquals(3, model.param3);
    assertFalse(model.wasWarmStarted);
  }

  @Test
  public void testClassificationExec() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Classifier) new DumbModel(), 5);
    instance.setUseWarmStarts(false);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.trainC(classData, ex);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedClassifier();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertFalse(model.wasWarmStarted);
  }

  @Test
  public void testClassificationWarm() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Classifier) new DumbModel(), 5);
    instance.setUseWarmStarts(true);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.trainC(classData);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedClassifier();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertTrue(model.wasWarmStarted);
  }

  @Test
  public void testClassificationWarmExec() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Classifier) new DumbModel(), 5);
    instance.setUseWarmStarts(true);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.trainC(classData, ex);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedClassifier();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertTrue(model.wasWarmStarted);
  }

  @Test
  public void testRegression() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Regressor) new DumbModel(), 5);
    instance.setUseWarmStarts(false);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.train(regData);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedRegressor();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertFalse(model.wasWarmStarted);
  }

  @Test
  public void testRegressionExec() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Regressor) new DumbModel(), 5);
    instance.setUseWarmStarts(false);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.train(regData, ex);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedRegressor();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertFalse(model.wasWarmStarted);
  }

  @Test
  public void testRegressionWarm() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Regressor) new DumbModel(), 5);
    instance.setUseWarmStarts(true);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.train(regData);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedRegressor();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertTrue(model.wasWarmStarted);
  }

  @Test
  public void testRegressionWarmExec() {
    System.out.println("setUseWarmStarts");
    GridSearch instance = new GridSearch((Regressor) new DumbModel(), 5);
    instance.setUseWarmStarts(true);

    instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
    instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
    instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);

    instance = instance.clone();
    instance.train(regData, ex);
    instance = instance.clone();

    final DumbModel model = (DumbModel) instance.getTrainedRegressor();
    assertEquals(1, model.param1);
    assertEquals(2, model.param2, 0.0);
    assertEquals(3, model.param3);
    assertTrue(model.wasWarmStarted);
  }
}
