package jsat.distributions.multivariate;

import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class NormalMTest
{
    Vec mean;
    Matrix covariance;
    
    public NormalMTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
        mean = DenseVector.toDenseVec(1, -1);
        covariance = new DenseMatrix(new double[][] 
        {
            {0.9, 0.4},
            {0.4, 0.3}
        });
    }

    /**
     * Test of setCovariance method, of class NormalM.
     */
    @Test
    public void testSetCovariance()
    {
        System.out.println("setCovariance");
        NormalM normalM = new NormalM(mean, covariance);
        
        Matrix t1 = Matrix.eye(3);//Should fail, too big
        Matrix t2 = new DenseMatrix(2, 3);//Should fail, not square
        Matrix t3 = new DenseMatrix(3, 2);//Should fail, not square
        
        Matrix[] shouldFail = new Matrix[] {t1, t2, t3};

        for (Matrix badMatrix : shouldFail)
            try
            {
                normalM.setCovariance(badMatrix);
                fail("Matrix was invalid, should have caused an exception");
            }
            catch (ArithmeticException ex)
            {
                //Good! Should fail
            }

    }

    /**
     * Test of logPdf method, of class NormalM.
     */
    @Test
    public void testLogPdf()
    {
        System.out.println("logPdf");
        NormalM normalM = new NormalM(mean, covariance);
        Vec[] xVals = new Vec[] 
        {
            DenseVector.toDenseVec(1, 0), DenseVector.toDenseVec(1, 1),
            DenseVector.toDenseVec(0, 1), DenseVector.toDenseVec(-1, 0),
            DenseVector.toDenseVec(0, -1), DenseVector.toDenseVec(-1, -1),
            DenseVector.toDenseVec(0, -11), DenseVector.toDenseVec(-11, -11),
            DenseVector.toDenseVec(1, -1), DenseVector.toDenseVec(-1, 1),
        };
        
        double[] pVals = new double[]
        {
            -4.825148700723577e+000, -1.709787597345085e+001,
            -2.573423960981450e+001, -1.755242142799632e+001,
            -2.097875973450849e+000, -6.188785064359943e+000,
            -3.748251487007237e+002, -1.698251487007236e+002,
            -7.342396098144846e-001, -3.709787597345088e+001,
        };

        for(int i = 0; i < pVals.length; i++)
            assertEquals(pVals[i], normalM.logPdf(xVals[i]), 1e-12);//Slightly smaller error b/c the absolute error can get large, but relative should still be good
    }

    /**
     * Test of pdf method, of class NormalM.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        NormalM normalM = new NormalM(mean, covariance);
        Vec[] xVals = new Vec[] 
        {
            DenseVector.toDenseVec(1, 0), DenseVector.toDenseVec(1, 1),
            DenseVector.toDenseVec(0, 1), DenseVector.toDenseVec(-1, 0),
            DenseVector.toDenseVec(0, -1), DenseVector.toDenseVec(-1, -1),
            DenseVector.toDenseVec(0, -11), DenseVector.toDenseVec(-11, -11),
            DenseVector.toDenseVec(1, -1), DenseVector.toDenseVec(-1, 1),
        };
        
        double[] pVals = new double[]
        {
            8.025360404763595e-003, 3.753935553147029e-008,
            6.664410523370948e-012, 2.382759609937137e-008,
            1.227168053837841e-001, 2.052318674310052e-003,
            1.642503261891859e-163, 1.761469107148436e-074,
            4.798702088783484e-001, 7.737437863769670e-017,
        };
        
        for(int i = 0; i < pVals.length; i++)
            assertEquals(pVals[i], normalM.pdf(xVals[i]), 1e-14);
    }
}
