/*
 * This implementation contributed under the Public Domain. 
 */
package jsat.linear;

import java.util.Arrays;
import java.util.Random;
import jsat.utils.random.RandomUtil;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author edwardraff
 */
public class TruncatedSVDTest {
    
    public TruncatedSVDTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void testSomeMethod()
    {
        Random rand = RandomUtil.getRandom(123);
        Matrix X_tall = DenseMatrix.random(100, 40, rand);
        Matrix X_wide = DenseMatrix.random(40, 100, rand);
        
        for(Matrix X : Arrays.asList(X_tall, X_wide))
        {
            double origNorm = X.frobenius();
            double prevNorm = X.frobenius();
            for(int k = 1; k < 40; k+= 4)
            {

                TruncatedSVD svd = new TruncatedSVD(X, k);

                Matrix U = svd.getU();
                Matrix V = svd.getV();

                Matrix R = U.clone();
                Matrix.diagMult(R, DenseVector.toDenseVec(svd.getSingularValues()));
                R = R.multiply(V);


                double cur_rec_cost = R.subtract(X).frobenius();
                assertTrue(cur_rec_cost < prevNorm);
                assertTrue(cur_rec_cost < origNorm);
                prevNorm = cur_rec_cost;
            }
        }
    }
    
}
