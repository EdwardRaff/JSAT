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
package jsat.math.optimization.stochastic;

import static org.junit.Assert.assertEquals;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.linear.DenseVector;
import jsat.linear.SubVector;
import jsat.linear.Vec;
import jsat.math.FunctionVec;
import jsat.math.optimization.RosenbrockFunction;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class AdamTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public AdamTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testUpdate_3args() {
    System.out.println("update");
    final Random rand = new XORWOW();
    final Vec x0 = new DenseVector(10);
    for (int i = 0; i < x0.length(); i++) {
      x0.set(i, rand.nextDouble());
    }

    final RosenbrockFunction f = new RosenbrockFunction();
    final FunctionVec fp = f.getDerivative();
    final double eta = 1.0;
    Adam instance = new Adam();
    instance.setup(x0.length());

    for (int i = 0; i < 100000; i++) {
      instance.update(x0, fp.f(x0), eta);
      instance = instance.clone();
    }
    assertEquals(0.0, f.f(x0), 1e-1);
  }

  @Test
  public void testUpdate_5args() {
    System.out.println("update");
    final Random rand = new XORWOW();
    final Vec xWithBias = new DenseVector(21);
    for (int i = 0; i < xWithBias.length(); i++) {
      xWithBias.set(i, rand.nextDouble());
    }

    final Vec x0 = new SubVector(0, 20, xWithBias);

    final RosenbrockFunction f = new RosenbrockFunction();
    final FunctionVec fp = f.getDerivative();
    final double eta = 1.0;

    Adam instance = new Adam();
    instance.setup(x0.length());

    for (int i = 0; i < 100000; i++) {
      final double bias = xWithBias.get(20);
      final Vec gradWithBias = fp.f(xWithBias);
      gradWithBias.normalize();
      final double biasGrad = gradWithBias.get(20);
      final Vec grad = new SubVector(0, 20, gradWithBias);
      final double biasDelta = instance.update(x0, grad, eta, bias, biasGrad);
      xWithBias.set(20, bias - biasDelta);

      instance = instance.clone();
    }
    assertEquals(0.0, f.f(xWithBias), 1e-1);

  }

}
