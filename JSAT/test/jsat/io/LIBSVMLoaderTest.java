/*
 * Copyright (C) 2015 Edward Raff
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
package jsat.io;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.utils.DoubleList;

/**
 *
 * @author Edward Raff
 */
public class LIBSVMLoaderTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public LIBSVMLoaderTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of loadR method, of class LIBSVMLoader.
   */
  @Test
  public void testLoadR_File() throws Exception {
    System.out.println("loadR");

    final List<String> testLines = new ArrayList<String>();
    final List<Double> expetedLabel = new DoubleList();
    final List<Vec> expectedVec = new ArrayList<Vec>();

    testLines.add("-1 2:3.0");// normal line
    expetedLabel.add(-1.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 3.0, 0.0, 0.0, 0.0));

    testLines.add("1 1:3.0 "); // line ends in a space
    expetedLabel.add(1.0);
    expectedVec.add(DenseVector.toDenseVec(3.0, 0.0, 0.0, 0.0, 0.0));

    testLines.add("-21 2:3.0 3:3.0 4:1.0");// normal line with many values
    expetedLabel.add(-21.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 3.0, 3.0, 1.0, 0.0));

    testLines.add("-1 2:3.0     4:2.0");// extra spaces in between
    expetedLabel.add(-1.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 3.0, 0.0, 2.0, 0.0));

    testLines.add("1"); /// empty line
    expetedLabel.add(1.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 0.0, 0.0, 0.0, 0.0));

    testLines.add("2 "); // empty line with space
    expetedLabel.add(2.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 0.0, 0.0, 0.0, 0.0));

    testLines.add("3");
    expetedLabel.add(3.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 0.0, 0.0, 0.0, 0.0));

    testLines.add("4");
    expetedLabel.add(4.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 0.0, 0.0, 0.0, 0.0));

    testLines.add("-1 1:10 3:2.0   "); // extra spaces at the end
    expetedLabel.add(-1.0);
    expectedVec.add(DenseVector.toDenseVec(10.0, 0.0, 2.0, 0.0, 0.0));

    testLines.add("2 2:3.0   3:3.0   5:1.0");// normal line with many values
    expetedLabel.add(2.0);
    expectedVec.add(DenseVector.toDenseVec(0.0, 3.0, 3.0, 0.0, 1.0));

    final String[] newLines = new String[] { "\n", "\n\r", "\r\n", "\n\r\n" };

    for (final boolean endInNewLines : new boolean[] { true, false }) {
      for (final String newLine : newLines) {
        for (int i = 0; i < testLines.size(); i++) {
          final StringBuilder input = new StringBuilder();
          for (int j = 0; j < i; j++) {
            input.append(testLines.get(j)).append(newLine);
          }
          input.append(testLines.get(i));
          if (endInNewLines) {
            input.append(newLine);
          }
          final RegressionDataSet dataSet = LIBSVMLoader.loadR(new StringReader(input.toString()), 0.5, 5);
          assertEquals(i + 1, dataSet.getSampleSize());
          for (int j = 0; j < i + 1; j++) {
            assertEquals(expetedLabel.get(j), dataSet.getTargetValue(j), 0.0);
            assertTrue(expectedVec.get(j).equals(dataSet.getDataPoint(j).getNumericalValues()));
          }
        }
      }
    }
  }

}
