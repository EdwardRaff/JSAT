package jsat.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class IntSetFixedSizeTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public IntSetFixedSizeTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of add method, of class IntSetFixedSize.
   */
  @Test
  public void testAdd_int() {
    System.out.println("add");
    final IntSetFixedSize set = new IntSetFixedSize(5);
    assertTrue(set.add(1));
    assertTrue(set.add(2));
    assertFalse(set.add(1));
    assertTrue(set.add(3));

    for (final int badVal : Arrays.asList(-1, 5, 10, -23)) {
      try {
        set.add(badVal);
        fail("Should not have been added");
      } catch (final Exception ex) {
      }
    }
  }

  /**
   * Test of add method, of class IntSetFixedSize.
   */
  @Test
  public void testAdd_Integer() {
    System.out.println("add");
    final IntSetFixedSize set = new IntSetFixedSize(5);
    assertTrue(set.add(new Integer(1)));
    assertTrue(set.add(new Integer(2)));
    assertFalse(set.add(new Integer(1)));
    assertTrue(set.add(new Integer(3)));

    for (final int badVal : Arrays.asList(-1, 5, 10, -23)) {
      try {
        set.add(new Integer(badVal));
        fail("Should not have been added");
      } catch (final Exception ex) {

      }
    }
  }

  /**
   * Test of contains method, of class IntSetFixedSize.
   */
  @Test
  public void testContains_int() {
    System.out.println("contains");
    final IntSetFixedSize instance = new IntSetFixedSize(100);
    final List<Integer> intList = new IntList();
    ListUtils.addRange(intList, 0, 100, 1);
    Collections.shuffle(intList);

    instance.addAll(intList.subList(0, 50));
    for (final int i : intList.subList(0, 50)) {
      assertTrue(instance.contains(i));
    }
    for (final int i : intList.subList(50, 100)) {
      assertFalse(instance.contains(i));
    }
  }

  /**
   * Test of contains method, of class IntSetFixedSize.
   */
  @Test
  public void testContains_Object() {
    System.out.println("contains");
    final IntSetFixedSize instance = new IntSetFixedSize(100);
    final List<Integer> intList = new IntList();
    ListUtils.addRange(intList, 0, 100, 1);
    Collections.shuffle(intList);

    instance.addAll(intList.subList(0, 50));
    for (final int i : intList.subList(0, 50)) {
      assertTrue(instance.contains(i));
    }
    for (final int i : intList.subList(50, 100)) {
      assertFalse(instance.contains(i));
    }

  }

  /**
   * Test of iterator method, of class IntSetFixedSize.
   */
  @Test
  public void testIterator() {
    System.out.println("iterator");
    final IntSetFixedSize set = new IntSetFixedSize(10);
    set.add(5);
    set.add(3);
    set.add(4);
    set.add(1);
    set.add(2);
    final Set<Integer> copySet = new IntSet(set);
    final int prev = Integer.MIN_VALUE;
    Iterator<Integer> iter = set.iterator();
    int count = 0;
    while (iter.hasNext()) {
      final int val = iter.next();
      copySet.remove(val);
      count++;
    }
    assertEquals(5, set.size());
    assertEquals(5, count);
    assertEquals(0, copySet.size());

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
   * Test of size method, of class IntSetFixedSize.
   */
  @Test
  public void testSize() {
    System.out.println("size");
    final IntSetFixedSize set = new IntSetFixedSize(10);
    assertEquals(0, set.size());
    set.add(1);
    assertEquals(1, set.size());
    set.add(1);
    set.add(2);
    assertEquals(2, set.size());
    set.add(5);
    set.add(7);
    set.add(2);
    assertEquals(4, set.size());
  }
}
