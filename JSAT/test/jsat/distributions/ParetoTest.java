package jsat.distributions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class ParetoTest {

  @BeforeClass
  public static void setUpBeforeClass() throws Exception {
  }

  @AfterClass
  public static void tearDownAfterClass() throws Exception {
  }

  @Before
  public void setUp() throws Exception {
  }

  @After
  public void tearDown() throws Exception {
  }

  @Test
  public void testEquals() {
    System.out.println("equals");
    final ContinuousDistribution d1 = new Pareto(0.5, 0.5);
    final ContinuousDistribution d2 = new Pareto(0.6, 0.5);
    final ContinuousDistribution d3 = new Pareto(0.5, 0.6);
    final ContinuousDistribution d4 = new Pareto(0.5, 0.5);
    final Integer i = 1;
    assertFalse(d1.equals(d2));
    assertFalse(d1.equals(d3));
    assertFalse(d2.equals(d3));
    assertFalse(d1.equals(i));
    assertFalse(d1.equals(null));
    assertEquals(d1, d1);
    assertEquals(d1, d4);
    assertEquals(d1, d1.clone());
  }

  @Test
  public void testHashCode() {
    System.out.println("hashCode");
    final ContinuousDistribution d1 = new Pareto(0.5, 0.5);
    final ContinuousDistribution d2 = new Pareto(0.6, 0.5);
    final ContinuousDistribution d4 = new Pareto(0.5, 0.5);
    assertEquals(d1.hashCode(), d4.hashCode());
    assertFalse(d1.hashCode() == d2.hashCode());
  }

}
