package jsat.utils;

import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class QuickSortTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public QuickSortTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of sort method, of class QuickSort.
   */
  @Test
  public void testSortD() {
    System.out.println("sort");
    final Random rand = new XORWOW();
    for (int size = 2; size < 10000; size *= 2) {
      final double[] x = new double[size];
      for (int i = 0; i < x.length; i++) {
        if (rand.nextInt(10) == 0) {
          x[i] = rand.nextInt(10);
        } else {
          x[i] = rand.nextDouble();
        }
      }
      QuickSort.sort(x, 0, size);

      for (int i = 0; i < x.length - 1; i++) {
        assertTrue(x[i] <= x[i + 1]);
      }
    }
  }

  @Test
  public void testSortDP() {
    System.out.println("sort");

    final IntList ints = new IntList();
    final Collection<List<?>> paired = new ArrayList<List<?>>();
    paired.add(ints);
    for (int size = 2; size < 10000; size *= 2) {
      ints.clear();
      final double[] x = new double[size];
      for (int i = 0; i < x.length; i++) {
        x[i] = x.length - i;
        ints.add(i);
      }
      QuickSort.sort(x, 0, size, paired);

      for (int i = 0; i < x.length - 1; i++) {
        assertTrue(x[i] <= x[i + 1]);
        assertTrue(ints.get(i) > ints.get(i + 1));
      }

    }
  }

  @Test
  public void testSortFP() {
    System.out.println("sort");

    final IntList ints = new IntList();
    final Collection<List<?>> paired = new ArrayList<List<?>>();
    paired.add(ints);
    for (int size = 2; size < 10000; size *= 2) {
      ints.clear();
      final float[] x = new float[size];
      for (int i = 0; i < x.length; i++) {
        x[i] = x.length - i;
        ints.add(i);
      }
      QuickSort.sort(x, 0, size, paired);

      for (int i = 0; i < x.length - 1; i++) {
        assertTrue(x[i] <= x[i + 1]);
        assertTrue(ints.get(i) > ints.get(i + 1));
      }

    }
  }

}
