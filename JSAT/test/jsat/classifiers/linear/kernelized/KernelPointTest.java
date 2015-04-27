package jsat.classifiers.linear.kernelized;

import jsat.distributions.kernels.KernelPoint;
import static java.lang.Math.*;
import java.util.List;
import java.util.Random;
import jsat.distributions.kernels.LinearKernel;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.*;
import jsat.linear.distancemetrics.EuclideanDistance;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class KernelPointTest
{
    List<Vec> toAdd;
    List<Vec> toTest;
    double[] coeff;
    
    public KernelPointTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
    }
    
    @AfterClass
    public static void tearDownClass()
    {
    }
    
    @Before
    public void setUp()
    {
        Vec mean = new DenseVector(new double[]{2.0, -1.0, 3.0});
        
        Matrix cov = new DenseMatrix(new double[][]
        {
            {1.07142,   1.15924,   0.38842},
            {1.15924,   1.33071,   0.51373},
            {0.38842,   0.51373,   0.92986},
        });
        
        NormalM normal = new NormalM(mean, cov);
        
        Random rand = new Random(42);
        toAdd = normal.sample(10, rand);
        toTest = normal.sample(10, rand);
        coeff = new double[toAdd.size()];
        for(int i = 0; i < coeff.length; i++)
            coeff[i] = Math.round(rand.nextDouble()*9+0.5);
        for(int i = 0; i < coeff.length; i++)
            if(i % 2 != 0)
                coeff[i] *= -1;
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of getSqrdNorm method, of class KernelPoint.
     */
    @Test
    public void testGetSqrdNorm()
    {
        System.out.println("getSqrdNorm");
        KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
        KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);
        
        for(int i = 0; i < toAdd.size(); i++)
        {
            Vec sumSimple = toAdd.get(0).clone();
            Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
            for(int ii = 1; ii < i+1; ii++ )
            {
                sumSimple.mutableAdd(toAdd.get(ii));
                sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
            }
            kpSimple.mutableAdd(toAdd.get(i));
            kpCoeff.mutableAdd(coeff[i], toAdd.get(i));
            
            double expectedSimple = Math.pow(sumSimple.pNorm(2), 2);
            double expectedCoeff = Math.pow(sumCoeff.pNorm(2), 2);
            
            assertEquals(expectedSimple, kpSimple.getSqrdNorm(), 1e-2*4);
            assertEquals(expectedCoeff, kpCoeff.getSqrdNorm(), 1e-2*4);
            
            KernelPoint kp0 = kpSimple.clone();
            KernelPoint kp1 = kpCoeff.clone();
            
            for(int j = i+1; j < coeff.length; j++ )
            {
                kp0.mutableAdd(toAdd.get(j));
                kp1.mutableAdd(coeff[j], toAdd.get(j));
            }
            
            for(int j = i+1; j < coeff.length; j++ )
            {
                kp0.mutableAdd(-1, toAdd.get(j));
                kp1.mutableAdd(-coeff[j], toAdd.get(j));
            }
            
            assertEquals(expectedSimple, kp0.getSqrdNorm(), 1e-2*4);
            assertEquals(expectedCoeff, kp1.getSqrdNorm(), 1e-2*4);
            
            kp0.mutableMultiply(1.0/(i+1));
            kp1.mutableMultiply(1.0/(i+1));
            
            assertEquals(expectedSimple/pow(i+1,2), kp0.getSqrdNorm(), 1e-2*4);
            assertEquals(expectedCoeff/pow(i+1,2), kp1.getSqrdNorm(), 1e-2*4);
        }
        
    }

    @Test
    public void testDot_KernelPoint()
    {
        System.out.println("dot_KernelPoint");
        KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
        KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);
        
        for(int i = 0; i < toAdd.size(); i++)
        {
            Vec sumSimple = toAdd.get(0).clone();
            Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
            for(int ii = 1; ii < i+1; ii++ )
            {
                sumSimple.mutableAdd(toAdd.get(ii));
                sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
            }
            kpSimple.mutableAdd(toAdd.get(i));
            kpCoeff.mutableAdd(coeff[i], toAdd.get(i));
            
            double expectedSimple = sumSimple.dot(sumSimple);
            double expectedCoeff = sumCoeff.dot(sumCoeff);
            double expectedSC = sumSimple.dot(sumCoeff);
            
            assertEquals(expectedSimple, kpSimple.dot(kpSimple), 1e-2*4);
            assertEquals(expectedCoeff, kpCoeff.dot(kpCoeff), 1e-2*4);
            assertEquals(expectedSC, kpSimple.dot(kpCoeff), 1e-2*4);
            
            KernelPoint kp0 = kpSimple.clone();
            KernelPoint kp1 = kpCoeff.clone();
            
            for(int j = i+1; j < coeff.length; j++ )
            {
                kp0.mutableAdd(toAdd.get(j));
                kp1.mutableAdd(coeff[j], toAdd.get(j));
            }
            
            for(int j = i+1; j < coeff.length; j++ )
            {
                kp0.mutableAdd(-1, toAdd.get(j));
                kp1.mutableAdd(-coeff[j], toAdd.get(j));
            }
            
            assertEquals(expectedSimple, kp0.dot(kpSimple), 1e-2*4);
            assertEquals(expectedCoeff, kp1.dot(kpCoeff), 1e-2*4);
            
            assertEquals(expectedSC, kp0.dot(kp1), 1e-2*4);
            assertEquals(expectedSC, kp1.dot(kp0), 1e-2*4);
            assertEquals(expectedSC, kp0.dot(kpCoeff), 1e-2*4);
            assertEquals(expectedSC, kpSimple.dot(kp1), 1e-2*4);
            
            kp0.mutableMultiply(1.0/(i+1));
            kp1.mutableMultiply(1.0/(i+1));
            
            assertEquals(expectedSimple/(i+1), kp0.dot(kpSimple), 1e-2*4);
            assertEquals(expectedCoeff/(i+1), kp1.dot(kpCoeff), 1e-2*4);
            
            assertEquals(expectedSC/pow(i+1, 2), kp0.dot(kp1), 1e-2*4);
            assertEquals(expectedSC/pow(i+1, 2), kp1.dot(kp0), 1e-2*4);
            assertEquals(expectedSC/(i+1), kp0.dot(kpCoeff), 1e-2*4);
            assertEquals(expectedSC/(i+1), kpSimple.dot(kp1), 1e-2*4);
        }
        
    }
    
    /**
     * Test of dot method, of class KernelPoint.
     */
    @Test
    public void testDot_Vec()
    {
        System.out.println("dot_Vec");
        KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
        KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);
        
        for(int i = 0; i < toAdd.size(); i++)
        {
            Vec sumSimple = toAdd.get(0).clone();
            Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
            for(int ii = 1; ii < i+1; ii++ )
            {
                sumSimple.mutableAdd(toAdd.get(ii));
                sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
            }
            kpSimple.mutableAdd(toAdd.get(i));
            kpCoeff.mutableAdd(coeff[i], toAdd.get(i));
            
            for(Vec v : toTest)
            {
                double expectedSimple = sumSimple.dot(v);
                double expectedCoeff = sumCoeff.dot(v);

                assertEquals(expectedSimple, kpSimple.dot(v), 1e-2*4);
                assertEquals(expectedCoeff, kpCoeff.dot(v), 1e-2*4);

                KernelPoint kp0 = kpSimple.clone();
                KernelPoint kp1 = kpCoeff.clone();

                for(int j = i+1; j < coeff.length; j++ )
                {
                    kp0.mutableAdd(toAdd.get(j));
                    kp1.mutableAdd(coeff[j], toAdd.get(j));
                }

                for(int j = i+1; j < coeff.length; j++ )
                {
                    kp0.mutableAdd(-1, toAdd.get(j));
                    kp1.mutableAdd(-coeff[j], toAdd.get(j));
                }

                assertEquals(expectedSimple, kp0.dot(v), 1e-2*4);
                assertEquals(expectedCoeff, kp1.dot(v), 1e-2*4);

                kp0.mutableMultiply(1.0/(i+1));
                kp1.mutableMultiply(1.0/(i+1));

                assertEquals(expectedSimple/(i+1), kp0.dot(v), 1e-2*4);
                assertEquals(expectedCoeff/(i+1), kp1.dot(v), 1e-2*4);
            }
        }
    }

    /**
     * Test of dist method, of class KernelPoint.
     */
    @Test
    public void testDistance_Vec()
    {
        System.out.println("distance_Vec");
        KernelPoint kpSimple = new KernelPoint(new LinearKernel(0), 1e-2);
        KernelPoint kpCoeff = new KernelPoint(new LinearKernel(0), 1e-2);
        
        EuclideanDistance d = new EuclideanDistance();
        
        for(int i = 0; i < toAdd.size(); i++)
        {
            Vec sumSimple = toAdd.get(0).clone();
            Vec sumCoeff = toAdd.get(0).multiply(coeff[0]);
            for(int ii = 1; ii < i+1; ii++ )
            {
                sumSimple.mutableAdd(toAdd.get(ii));
                sumCoeff.mutableAdd(coeff[ii], toAdd.get(ii));
            }
            kpSimple.mutableAdd(toAdd.get(i));
            kpCoeff.mutableAdd(coeff[i], toAdd.get(i));
            
            for(Vec v : toTest)
            {
                double expectedSimple = d.dist(sumSimple, v);
                double expectedCoeff = d.dist(sumCoeff, v);

                assertEquals(expectedSimple, kpSimple.dist(v), 1e-2*4);
                assertEquals(expectedCoeff, kpCoeff.dist(v), 1e-2*4);

                KernelPoint kp0 = kpSimple.clone();
                KernelPoint kp1 = kpCoeff.clone();

                for(int j = i+1; j < coeff.length; j++ )
                {
                    kp0.mutableAdd(toAdd.get(j));
                    kp1.mutableAdd(coeff[j], toAdd.get(j));
                }

                for(int j = i+1; j < coeff.length; j++ )
                {
                    kp0.mutableAdd(-1, toAdd.get(j));
                    kp1.mutableAdd(-coeff[j], toAdd.get(j));
                }

                assertEquals(expectedSimple, kp0.dist(v), 1e-2*4);
                assertEquals(expectedCoeff, kp1.dist(v), 1e-2*4);

                kp0.mutableMultiply(1.0/(i+1));
                kp1.mutableMultiply(1.0/(i+1));

                assertEquals(d.dist(sumSimple.divide(i+1), v), kp0.dist(v), 1e-2*4);
                assertEquals(d.dist(sumCoeff.divide(i+1), v), kp1.dist(v), 1e-2*4);
            }
        }
    }
    
    @Test
    public void testDistance_KernelPoint()
    {
        System.out.println("distance_KernelPoint");
        KernelPoint kpSimpleA = new KernelPoint(new LinearKernel(0), 1e-2);
        KernelPoint kpCoeffA = new KernelPoint(new LinearKernel(0), 1e-2);
        
        KernelPoint kpSimpleB = new KernelPoint(new LinearKernel(0), 1e-2);
        KernelPoint kpCoeffB = new KernelPoint(new LinearKernel(0), 1e-2);
        
        EuclideanDistance d = new EuclideanDistance();
        
        for(int i = 0; i < toAdd.size(); i++)
        {
            Vec sumSimpleA = toAdd.get(0).clone();
            Vec sumCoeffA = toAdd.get(0).multiply(coeff[0]);
            for(int ii = 1; ii < i+1; ii++ )
            {
                sumSimpleA.mutableAdd(toAdd.get(ii));
                sumCoeffA.mutableAdd(coeff[ii], toAdd.get(ii));
            }
            
            Vec sumSimpleB = toTest.get(0).clone();
            Vec sumCoeffB = toTest.get(0).multiply(coeff[0]);
            for(int ii = 1; ii < i+1; ii++ )
            {
                sumSimpleB.mutableAdd(toTest.get(ii));
                sumCoeffB.mutableAdd(coeff[ii], toTest.get(ii));
            }
            
            kpSimpleA.mutableAdd(toAdd.get(i));
            kpCoeffA.mutableAdd(coeff[i], toAdd.get(i));
            
            kpSimpleB.mutableAdd(toTest.get(i));
            kpCoeffB.mutableAdd(coeff[i], toTest.get(i));
            
            assertEquals(0.0, kpSimpleA.dist(kpSimpleA), 1e-2*4);
            assertEquals(0.0, kpSimpleB.dist(kpSimpleB), 1e-2*4);
            assertEquals(0.0, kpCoeffA.dist(kpCoeffA), 1e-2*4);
            assertEquals(0.0, kpCoeffB.dist(kpCoeffB), 1e-2*4);
            
            
            assertEquals(d.dist(sumSimpleA, sumSimpleB), kpSimpleA.dist(kpSimpleB), 1e-2*4);
            assertEquals(d.dist(sumSimpleA, sumCoeffA), kpSimpleA.dist(kpCoeffA), 1e-2*4);
            assertEquals(d.dist(sumSimpleA, sumCoeffB), kpSimpleA.dist(kpCoeffB), 1e-2*4);
            
            assertEquals(d.dist(sumCoeffA, sumSimpleB), kpCoeffA.dist(kpSimpleB), 1e-2*4);
            assertEquals(d.dist(sumCoeffB, sumSimpleB), kpCoeffB.dist(kpSimpleB), 1e-2*4);
            
            KernelPoint kpSimpleAClone = kpSimpleA.clone();
            KernelPoint kpSimpleBClone = kpSimpleB.clone();
            kpSimpleAClone.mutableMultiply(1.0/(i+1));
            kpSimpleBClone.mutableMultiply(1.0/(i+1));
            
            
            assertEquals(d.dist(sumSimpleA.divide(i+1), sumSimpleB.divide(i+1)), kpSimpleAClone.dist(kpSimpleBClone), 1e-2*4);
            assertEquals(d.dist(sumSimpleA.divide(i+1), sumCoeffA), kpSimpleAClone.dist(kpCoeffA), 1e-2*4);
            assertEquals(d.dist(sumSimpleA.divide(i+1), sumCoeffB), kpSimpleAClone.dist(kpCoeffB), 1e-2*4);
            
            assertEquals(d.dist(sumCoeffA, sumSimpleB.divide(i+1)), kpCoeffA.dist(kpSimpleBClone), 1e-2*4);
            assertEquals(d.dist(sumCoeffB, sumSimpleB.divide(i+1)), kpCoeffB.dist(kpSimpleBClone), 1e-2*4);
        }
    }
}
