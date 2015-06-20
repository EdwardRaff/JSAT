package jsat.distributions.empirical;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import jsat.distributions.ContinuousDistribution;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.distributions.empirical.kernelfunc.UniformKF;
import jsat.linear.DenseVector;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class KernelDensityEstimatorTest {
	static List<ContinuousDistribution> unequal = new ArrayList<ContinuousDistribution>();
	static ContinuousDistribution d1;
	static ContinuousDistribution d2;
	static ContinuousDistribution d3;
	static ContinuousDistribution d4;

	private static final int vecSize = 1000;
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		Random r1 = new Random(5079392926462355615L);
		Random r2 = new Random(7628304882101574242L);
		double[] vec1 = new double[vecSize];
		double[] vec2 = new double[vecSize];
		double[] vec3 = new double[vecSize];
		for(int i = 0;i<vecSize;i++){
			double temp = r1.nextDouble();
			vec1[i]=temp;
			vec2[i]=r2.nextDouble();
			vec3[i]=temp;
		}
		d1 = new KernelDensityEstimator(new DenseVector(vec1));
		d2 = new KernelDensityEstimator(new DenseVector(vec3));
		
		d3 = new KernelDensityEstimator(new DenseVector(vec1), GaussKF.getInstance(), 2, vec1);
		d4 = new KernelDensityEstimator(new DenseVector(vec1), GaussKF.getInstance(), 2, vec1);
		
		unequal.add(d1);
		unequal.add(d3);
		unequal.add(new KernelDensityEstimator(new DenseVector(vec2)));
		unequal.add(new KernelDensityEstimator(new DenseVector(vec1), GaussKF.getInstance(), 1, vec1));
		unequal.add(new KernelDensityEstimator(new DenseVector(vec1), UniformKF.getInstance(), 2, vec1));
		unequal.add(new KernelDensityEstimator(new DenseVector(vec1), GaussKF.getInstance(), 2, vec2));
		unequal.add(new KernelDensityEstimator(new DenseVector(vec2), GaussKF.getInstance(), 2, vec1));
		unequal.add(new KernelDensityEstimator(new DenseVector(new double[]{1,3,5})));
		unequal.add(new KernelDensityEstimator(new DenseVector(new double[]{0,3,6})));
		unequal.add(new KernelDensityEstimator(new DenseVector(new double[]{1,1,1})));
		unequal.add(new KernelDensityEstimator(new DenseVector(new double[]{1,1,1,1,1})));
		unequal.add(new KernelDensityEstimator(new DenseVector(new double[]{2,2,2,2,3,4,4,4,4})));
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
    	Integer k = new Integer(1);
    	assertFalse(d1.equals(k));
    	assertFalse(d1.equals(null));
    	
    	assertEquals(d1, d1);
    	assertEquals(d1, d2);
    	assertEquals(d3, d4);
    	
    	assertEquals(d1, d1.clone());
    	
    	for(int i = 0;i<unequal.size();i++){
    		for(int j = 0;j<unequal.size();j++){
    			if(i!=j){
    				assertFalse(unequal.get(i).equals(unequal.get(j)));
    			}
    		}
    	}
    }
    
    @Test
    public void testHashCode(){
    	System.out.println("hashCode");
    	assertEquals(d1.hashCode(), d2.hashCode());
    	assertFalse(d1.hashCode()==d3.hashCode());
    }

}
