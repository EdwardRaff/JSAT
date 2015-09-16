/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
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
public class LongDoubleMapTest {

  private static final int TEST_SIZE = 2000;

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  Random rand;

  public LongDoubleMapTest() {
  }

  private void assertEntriesAreEqual(final Map<Long, Double> truthMap, final LongDoubleMap ldMap) {
    assertEquals(truthMap.size(), ldMap.size());

    final Map<Long, Double> copy = new HashMap<Long, Double>();

    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      assertEquals(entry.getValue(), ldMap.get(entry.getKey()));
    }

    int observed = 0;
    for (final Entry<Long, Double> entry : ldMap.entrySet()) {
      copy.put(entry.getKey(), entry.getValue());
      observed++;
      assertTrue(truthMap.containsKey(entry.getKey()));
      assertEquals(truthMap.get(entry.getKey()), entry.getValue());
    }
    assertEquals(truthMap.size(), observed);

    // make sure we put every value into the copy!
    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      assertEquals(truthMap.get(entry.getKey()), copy.get(entry.getKey()));
    }
  }

  private void removeEvenByIterator(final Iterator<Entry<Long, Double>> iterator) {
    while (iterator.hasNext()) {
      final Entry<Long, Double> entry = iterator.next();
      if (entry.getKey() % 2 == 0) {
        iterator.remove();
      }
    }
  }

  @Before
  public void setUp() {
    rand = new XORWOW();
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of containsKey method, of class LongDoubleMap.
   */
  @Test
  public void testContainsKey_long() {
    System.out.println("containsKey");
    Long key = null;
    Double value = null;

    final Map<Long, Double> truthMap = new HashMap<Long, Double>();
    final LongDoubleMap ldMap = new LongDoubleMap();

    final int MAX = TEST_SIZE / 2;
    for (int i = 0; i < MAX; i++) {
      key = Long.valueOf(rand.nextInt(MAX));
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Long keyInSet : truthMap.keySet()) {
      assertTrue(ldMap.containsKey(keyInSet.longValue()));
    }

    for (long i = MAX + 1; i < MAX * 2; i++) {
      assertFalse(ldMap.containsKey(i));
    }
  }

  /**
   * Test of containsKey method, of class LongDoubleMap.
   */
  @Test
  public void testContainsKey_Object() {
    System.out.println("containsKey");
    Long key = null;
    Double value = null;

    final Map<Long, Double> truthMap = new HashMap<Long, Double>();
    final LongDoubleMap ldMap = new LongDoubleMap();

    final int MAX = TEST_SIZE / 2;
    for (int i = 0; i < MAX; i++) {
      key = Long.valueOf(rand.nextInt(MAX));
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Long keyInSet : truthMap.keySet()) {
      assertTrue(ldMap.containsKey(keyInSet));
    }

    for (long i = MAX + 1; i < MAX * 2; i++) {
      assertFalse(ldMap.containsKey(Long.valueOf(i)));
    }
  }

  /**
   * Test of increment method, of class LongDoubleMap.
   */
  @Test
  public void testIncrement() {
    System.out.println("increment");
    Long key = null;
    Double value = null;

    final Map<Long, Double> truthMap = new HashMap<Long, Double>();
    final LongDoubleMap ldMap = new LongDoubleMap();

    final int MAX = TEST_SIZE / 2;
    int times = 0;
    for (int i = 0; i < MAX; i++) {
      key = Long.valueOf(rand.nextInt(MAX));
      value = Double.valueOf(rand.nextInt(1000));
      if (truthMap.containsKey(key)) {
        times++;
      }
      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);

      if (prev == null && prevTruth != null) {
        System.out.println(ldMap.put(key, value));
      }
      assertEquals(prevTruth, prev);
      if (ldMap.size() != truthMap.size()) {
        System.out.println();
      }
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      final double delta = Double.valueOf(rand.nextInt(100));
      final double trueNewValue = entry.getValue() + delta;
      entry.setValue(trueNewValue);
      final double newValue = ldMap.increment(entry.getKey(), delta);
      assertEquals(trueNewValue, newValue, 0.0);
    }

    for (int i = MAX; i < MAX * 2; i++) {
      key = Long.valueOf(i);// force it to be new
      value = Double.valueOf(rand.nextInt(1000));

      truthMap.put(key, value);
      final double ldNew = ldMap.increment(key, value);
      assertEquals(value, ldNew, 0.0);
    }

    assertEntriesAreEqual(truthMap, ldMap);
  }

  /**
   * Test of put method, of class LongDoubleMap.
   */
  @Test
  public void testPut_long_double() {
    System.out.println("put");
    long key;
    double value;

    Map<Long, Double> truthMap = new HashMap<Long, Double>();
    LongDoubleMap ldMap = new LongDoubleMap();

    for (int i = 0; i < TEST_SIZE; i++) {
      key = rand.nextLong();
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      Double prev = ldMap.put(key, value);
      if (prev.isNaN()) {
        prev = null;
      }
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    // will call the iterator remove on everythin
    removeEvenByIterator(ldMap.entrySet().iterator());
    removeEvenByIterator(truthMap.entrySet().iterator());

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Entry<Long, Double> entry : ldMap.entrySet()) {
      entry.setValue(1.0);
    }
    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      entry.setValue(1.0);
    }

    assertEntriesAreEqual(truthMap, ldMap);

    /// again, random keys - and make them colide
    truthMap = new HashMap<Long, Double>();
    ldMap = new LongDoubleMap();

    for (int i = 0; i < TEST_SIZE; i++) {
      key = Long.valueOf(rand.nextInt(50000));
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      Double prev = ldMap.put(key, value);
      if (prev.isNaN()) {
        prev = null;
      }
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    // will call the iterator remove on everythin
    removeEvenByIterator(ldMap.entrySet().iterator());
    removeEvenByIterator(truthMap.entrySet().iterator());

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Entry<Long, Double> entry : ldMap.entrySet()) {
      entry.setValue(1.0);
    }
    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      entry.setValue(1.0);
    }

    assertEntriesAreEqual(truthMap, ldMap);
  }

  /**
   * Test of put method, of class LongDoubleMap.
   */
  @Test
  public void testPut_Long_Double() {
    System.out.println("put");
    Long key = null;
    Double value = null;

    Map<Long, Double> truthMap = new HashMap<Long, Double>();
    LongDoubleMap ldMap = new LongDoubleMap();

    for (int i = 0; i < TEST_SIZE; i++) {
      key = rand.nextLong();
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    // will call the iterator remove on everythin
    removeEvenByIterator(ldMap.entrySet().iterator());
    removeEvenByIterator(truthMap.entrySet().iterator());

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Entry<Long, Double> entry : ldMap.entrySet()) {
      entry.setValue(1.0);
    }
    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      entry.setValue(1.0);
    }

    assertEntriesAreEqual(truthMap, ldMap);

    /// again, random keys - and make them colide
    truthMap = new HashMap<Long, Double>();
    ldMap = new LongDoubleMap();

    for (int i = 0; i < TEST_SIZE; i++) {
      key = Long.valueOf(rand.nextInt(50000));
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    // will call the iterator remove on everythin
    removeEvenByIterator(ldMap.entrySet().iterator());
    removeEvenByIterator(truthMap.entrySet().iterator());

    assertEntriesAreEqual(truthMap, ldMap);

    for (final Entry<Long, Double> entry : ldMap.entrySet()) {
      entry.setValue(1.0);
    }
    for (final Entry<Long, Double> entry : truthMap.entrySet()) {
      entry.setValue(1.0);
    }

    assertEntriesAreEqual(truthMap, ldMap);
  }

  /**
   * Test of remove method, of class LongDoubleMap.
   */
  @Test
  public void testRemove_long() {
    System.out.println("remove");
    Long key = null;
    Double value = null;

    final Map<Long, Double> truthMap = new HashMap<Long, Double>();
    final LongDoubleMap ldMap = new LongDoubleMap();

    final int MAX = TEST_SIZE / 2;
    for (int i = 0; i < MAX; i++) {
      key = Long.valueOf(rand.nextInt(MAX));
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    for (int i = 0; i < MAX / 4; i++) {
      key = Long.valueOf(rand.nextInt(MAX));

      final Double prevTruth = truthMap.remove(key);
      Double prev = ldMap.remove(key.longValue());
      if (prev.isNaN()) {
        prev = null;
      }
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);
  }

  /**
   * Test of remove method, of class LongDoubleMap.
   */
  @Test
  public void testRemove_Object() {
    System.out.println("remove");
    Long key = null;
    Double value = null;

    final Map<Long, Double> truthMap = new HashMap<Long, Double>();
    final LongDoubleMap ldMap = new LongDoubleMap();

    final int MAX = TEST_SIZE / 2;
    for (int i = 0; i < MAX; i++) {
      key = Long.valueOf(rand.nextInt(MAX));
      value = Double.valueOf(rand.nextInt(1000));

      final Double prevTruth = truthMap.put(key, value);
      final Double prev = ldMap.put(key, value);
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);

    for (int i = 0; i < MAX / 4; i++) {
      key = Long.valueOf(rand.nextInt(MAX));

      final Double prevTruth = truthMap.remove(key);
      Double prev = ldMap.remove(key);
      if (prevTruth == null && prev != null) {
        prev = ldMap.remove(key);
      }
      assertEquals(prevTruth, prev);
      assertEquals(truthMap.size(), ldMap.size());
    }

    assertEntriesAreEqual(truthMap, ldMap);
  }

}
