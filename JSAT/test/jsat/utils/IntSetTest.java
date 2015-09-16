/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class IntSetTest {

  @BeforeClass
  public static void setUpClass() throws Exception {
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
  }

  public IntSetTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of add method, of class IntSet.
   */
  @Test
  public void testAdd() {
    System.out.println("add");
    final IntSet set = new IntSet();
    assertFalse(set.add(null));
    assertTrue(set.add(1));
    assertTrue(set.add(2));
    assertFalse(set.add(1));
    assertFalse(set.add(null));
    assertTrue(set.add(3));
  }

  /**
   * Test of iterator method, of class IntSet.
   */
  @Test
  public void testIterator() {
    System.out.println("iterator");
    final IntSet set = new IntSet();
    set.add(5);
    set.add(3);
    set.add(4);
    set.add(1);
    set.add(2);
    int prev = Integer.MIN_VALUE;
    Iterator<Integer> iter = set.iterator();
    int count = 0;
    while (iter.hasNext()) {
      final int val = iter.next();
      count++;
      assertTrue(prev < val);
      prev = val;
    }
    assertEquals(5, set.size());
    assertEquals(5, count);

    // Test removing some elements
    iter = set.iterator();
    while (iter.hasNext()) {
      final int val = iter.next();
      if (val == 2 || val == 4) {
        iter.remove();
      }
    }
    assertEquals(3, set.size());

    // Make sure the corect values were actually removed
    iter = set.iterator();
    count = 0;
    while (iter.hasNext()) {
      final int val = iter.next();
      assertFalse(val == 2 || val == 4);
      count++;
    }
    assertEquals(3, set.size());
    assertEquals(3, count);
  }

  /**
   * Test of size method, of class IntSet.
   */
  @Test
  public void testSize() {
    System.out.println("size");
    final IntSet set = new IntSet();
    assertEquals(0, set.size());
    set.add(1);
    assertEquals(1, set.size());
    set.add(1);
    set.add(2);
    assertEquals(2, set.size());
    set.add(5);
    set.add(-4);
    set.add(2);
    assertEquals(4, set.size());
  }
}
