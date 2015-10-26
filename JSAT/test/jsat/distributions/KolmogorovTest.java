package jsat.distributions;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class KolmogorovTest {

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
    public void testEquals(){
    	System.out.println("equals");
    	final ContinuousDistribution d1 = new Kolmogorov();
    	final ContinuousDistribution d4 = new Kolmogorov();
    	final Integer i = new Integer(1);
    	assertFalse(d1.equals(i));
    	assertFalse(d1.equals(null));
    	assertEquals(d1, d1);
    	assertEquals(d1, d4);
    	assertEquals(d1, d1.clone());
    }
    
    @Test
    public void testHashCode(){
    	System.out.println("hashCode");
    	final ContinuousDistribution d1 = new Kolmogorov();
    	final ContinuousDistribution d4 = new Kolmogorov();
    	assertEquals(d1.hashCode(), d4.hashCode());
    }

}
