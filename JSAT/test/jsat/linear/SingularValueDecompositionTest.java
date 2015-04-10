/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.linear;

import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import jsat.utils.SystemInfo;
import java.util.concurrent.ExecutorService;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class SingularValueDecompositionTest
{
    /**
     * 5x5
     */
    static DenseMatrix A;
    /**
     * 5x5
     */
    static DenseMatrix B;
    
    /**
     * Rank 3 5x5
     */
    static DenseMatrix D;
    /**
     * 5x7
     */
    static DenseMatrix C;
    
    /**
     * 7 x 5
     */
    static DenseMatrix E;
    
    private static final double delta = 1e-10;
        
    static ExecutorService threadpool = Executors.newFixedThreadPool(SystemInfo.LogicalCores+1, new ThreadFactory() {

        public Thread newThread(Runnable r)
        {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            return thread;
        }
    });
    
    public SingularValueDecompositionTest()
    {
        A = new DenseMatrix(new double[][] 
        {
            {1, 5, 4, 8, 9},
            {1, 5, 7, 3, 7},
            {0, 3, 8, 5, 6},
            {3, 8, 0, 7, 0},
            {1, 9, 2, 9, 6}
        } );
        
        B = new DenseMatrix(new double[][] 
        {
            {5, 3, 2, 8, 8},
            {1, 8, 3, 6, 8},
            {1, 2, 6, 5, 4},
            {3, 9, 5, 9, 6},
            {8, 3, 4, 3, 1}
        } );
        
        C = new DenseMatrix(new double[][] 
        {
            {1, 6, 8, 3, 1, 5, 10},
            {5, 5, 3, 7, 2, 10, 0},
            {8, 0, 5, 7, 9, 1, 8},
            {9, 3, 2, 7, 2, 4, 8},
            {1, 2, 6, 5, 8, 1, 9}
        } );
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
        A = new DenseMatrix(new double[][] 
        {
            {1, 5, 4, 8, 9},
            {1, 5, 7, 3, 7},
            {0, 3, 8, 5, 6},
            {3, 8, 0, 7, 0},
            {1, 9, 2, 9, 6}
        } );
        
        B = new DenseMatrix(new double[][] 
        {
            {5, 3, 2, 8, 8},
            {1, 8, 3, 6, 8},
            {1, 2, 6, 5, 4},
            {3, 9, 5, 9, 6},
            {8, 3, 4, 3, 1}
        } );
        
        D = new DenseMatrix(new double[][] 
        {
            {5, 3, 2, 0, 0},
            {1, 8, 3, 0, 0},
            {1, 2, 6, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0}
        } );
        
        C = new DenseMatrix(new double[][] 
        {
            {1, 6, 8, 3, 1, 5, 10},
            {5, 5, 3, 7, 2, 10, 0},
            {8, 0, 5, 7, 9, 1, 8},
            {9, 3, 2, 7, 2, 4, 8},
            {1, 2, 6, 5, 8, 1, 9}
        } );
        
        E = new DenseMatrix(new double[][]
        {
            {4,     4,     4,     2,     4},
            {1,     5,     2,     3,     0},
            {4,     2,     3,     4,     0},
            {1,     9,     4,     2,     7},
            {5,     7,     4,     3,     5},
            {5,     9,     3,     0,     9},
            {5,     4,     5,     8,     4}
        });
        
    }

    /**
     * Test of getSingularValues method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetSingularValues()
    {
        System.out.println("getSingularValues");
        double[] sValsATrue = new double[] {25.615015549269760,   9.967372402268001 ,  4.046901102951370 ,  2.356215314072247,   1.262262517005518};
        double[] sValsDTrue = new double[] {10.866928472828468,   4.351234464538032,   3.997076957108316,                   0,                   0};
        double[] sValsCTrue = new double[] {29.846912916029009,  11.902860295602228,   9.905493706861000,   6.102122989264148,   1.768896722137177};
        
        double[] sValsA = new SingularValueDecomposition(A).getSingularValues();
        double[] sValsD = new SingularValueDecomposition(D).getSingularValues();
        double[] sValsC = new SingularValueDecomposition(C).getSingularValues();
        
        for(int i = 0; i < sValsA.length; i++)
        {
            assertEquals(sValsATrue[i], sValsA[i], delta);
            assertEquals(sValsCTrue[i], sValsC[i], delta);
            assertEquals(sValsDTrue[i], sValsD[i], delta);
        }
    }

    /**
     * Test of getNorm2 method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetNorm2()
    {
        System.out.println("getNorm2");
        double trueNormA = 25.615015549269760;
        double trueNormC = 29.846912916029009;
        double trueNormD = 10.866928472828468;
        
        assertEquals(trueNormA, new SingularValueDecomposition(A).getNorm2(), delta);
        assertEquals(trueNormC, new SingularValueDecomposition(C).getNorm2(), delta);
        assertEquals(trueNormD, new SingularValueDecomposition(D).getNorm2(), delta);
    }

    /**
     * Test of getCondition method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetCondition()
    {
        System.out.println("getCondition");
        double trueCondA = 20.292938437272621;
        double trueCondC = 16.873180068968665;
        double trueCondD = Double.POSITIVE_INFINITY;
        
        assertEquals(trueCondA, new SingularValueDecomposition(A).getCondition(), delta);
        assertEquals(trueCondC, new SingularValueDecomposition(C).getCondition(), delta);
        assertEquals(trueCondD, new SingularValueDecomposition(D).getCondition(), delta);
    }

    /**
     * Test of getRank method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetRank_0args()
    {
        System.out.println("getRank");
        int rankA = 5;
        int rankC = 5;
        int rankD = 3;
        
        assertEquals(rankA, new SingularValueDecomposition(A).getRank());
        assertEquals(rankC, new SingularValueDecomposition(C).getRank());
        assertEquals(rankD, new SingularValueDecomposition(D).getRank());
    }

    /**
     * Test of getRank method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetRank_double()
    {
        System.out.println("getRank");
        double tol = 5.0;//A very large tolerance!
        int rankA = 2;
        int rankC = 4;
        int rankD = 1;
        
        assertEquals(rankA, new SingularValueDecomposition(A).getRank(tol));
        assertEquals(rankC, new SingularValueDecomposition(C).getRank(tol));
        assertEquals(rankD, new SingularValueDecomposition(D).getRank(tol));
    }

    /**
     * Test of getInverseSingularValues method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetInverseSingularValues_0args()
    {
        System.out.println("getInverseSingularValues");
        double[] sValsATrue = new double[] {1.0/25.615015549269760,   1.0/9.967372402268001 ,  1.0/4.046901102951370 ,  1.0/2.356215314072247,   1.0/1.262262517005518};
        double[] sValsDTrue = new double[] {1.0/10.866928472828468,   1.0/4.351234464538032,   1.0/3.997076957108316,                   0,                   0};
        double[] sValsCTrue = new double[] {1.0/29.846912916029009,  1.0/11.902860295602228,   1.0/9.905493706861000,   1.0/6.102122989264148,   1.0/1.768896722137177};
        
        double[] sValsA = new SingularValueDecomposition(A).getInverseSingularValues();
        double[] sValsD = new SingularValueDecomposition(D).getInverseSingularValues();
        double[] sValsC = new SingularValueDecomposition(C).getInverseSingularValues();
        
        for(int i = 0; i < sValsA.length; i++)
        {
            assertEquals(sValsATrue[i], sValsA[i], delta);
            assertEquals(sValsCTrue[i], sValsC[i], delta);
            assertEquals(sValsDTrue[i], sValsD[i], delta);
        }
    }

    /**
     * Test of getInverseSingularValues method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetInverseSingularValues_double()
    {
        System.out.println("getInverseSingularValues");
        double tol = 5.0;
        double[] sValsATrue = new double[] {1.0/25.615015549269760,   1.0/9.967372402268001 ,  0 ,  0,   0};
        double[] sValsDTrue = new double[] {1.0/10.866928472828468,   0,   0,                   0,                   0};
        double[] sValsCTrue = new double[] {1.0/29.846912916029009,  1.0/11.902860295602228,   1.0/9.905493706861000,   1.0/6.102122989264148,   0};
        
        double[] sValsA = new SingularValueDecomposition(A).getInverseSingularValues(tol);
        double[] sValsD = new SingularValueDecomposition(D).getInverseSingularValues(tol);
        double[] sValsC = new SingularValueDecomposition(C).getInverseSingularValues(tol);
        
        for(int i = 0; i < sValsA.length; i++)
        {
            assertEquals(sValsATrue[i], sValsA[i], delta);
            assertEquals(sValsCTrue[i], sValsC[i], delta);
            assertEquals(sValsDTrue[i], sValsD[i], delta);
        }
    }

    /**
     * Test of getPseudoInverse method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetPseudoInverse()
    {
        System.out.println("getPseudoInverse");
        Matrix truePInvC = new DenseMatrix(new double[][]
        {
            {   0.062400933442643,  -0.016718531213984,   0.168007905973973,  -0.020034948098766,  -0.198766823807711 },
            {  -0.005194625607728,   0.020172975789774,  -0.094455454689009,   0.047276210963183,   0.067661938176502 },
            {   0.167355234385855,   0.005639718873814,   0.223102518285777,  -0.190852367999693,  -0.206282098318648 },
            {  -0.111390759232766,   0.034725918075815,  -0.129066299672170,   0.102432370984900,   0.160549895284892 },
            {  -0.060231213795913,   0.032065710226917,   0.023059509602175,  -0.051296309654649,   0.087923910856982 },
            {   0.011210049988905,   0.065859577351507,  -0.017972473278131,  -0.017808318790801,  -0.002532636559622 },
            {  -0.003172181333343,  -0.061497842572707,  -0.093209356832114,   0.109623741697116,   0.088614156738433 }
        });
        
        Matrix truePInvD = new DenseMatrix(new double[][]
        {
            {   0.222222222222222,  -0.074074074074074,  -0.037037037037037,                   0,                   0},
            {  -0.015873015873016,   0.148148148148148,  -0.068783068783069,                   0,                   0},
            {  -0.031746031746032,  -0.037037037037037,   0.195767195767196,                   0,                   0},
            {                   0,                   0,                   0,                   0,                   0},
            {                   0,                   0,                   0,                   0,                   0}
        });
        
        Matrix pInvC = new SingularValueDecomposition(C.clone()).getPseudoInverse();
        Matrix pInvD = new SingularValueDecomposition(D.clone()).getPseudoInverse();
        
        assertTrue(truePInvD.equals(pInvD, delta));
        assertTrue(truePInvC.equals(pInvC, delta));
        
    }

    /**
     * Test of getPseudoDet method, of class SingularValueDecomposition.
     */
    @Test
    public void testGetPseudoDet_0args()
    {
        System.out.println("getPseudoDet");
        double pDetA = 3073;
        double pDetB = 8068;
        double pDetC = 3.798484118697878e+004;
        double pDetD = 189;
        
        assertEquals(pDetA, new SingularValueDecomposition(A).getPseudoDet(), delta);
        assertEquals(pDetB, new SingularValueDecomposition(B).getPseudoDet(), delta);
        assertEquals(pDetC, new SingularValueDecomposition(C).getPseudoDet(), delta);
        assertEquals(pDetD, new SingularValueDecomposition(D).getPseudoDet(), delta);
        
    }

    /**
     * Test of solve method, of class SingularValueDecomposition.
     */
    @Test
    public void testSolve_Vec()
    {
        System.out.println("solve");
        Vec b = DenseVector.toDenseVec(1.0, 2.0, 3.0, 4.0, 5.0);
        SingularValueDecomposition instance = null;
        
        instance = new SingularValueDecomposition(A.clone());
        Vec x = instance.solve(b);
        assertTrue(A.multiply(x).equals(b, delta));
        
        instance = new SingularValueDecomposition(C.transpose());
        try
        {
            x = instance.solve(b);
            fail("Should not have occured");
        }
        catch(ArithmeticException ex)
        {
            
        }
        
    }

    /**
     * Test of solve method, of class SingularValueDecomposition.
     */
    @Test
    public void testSolve_Matrix()
    {
        System.out.println("solve");
        SingularValueDecomposition instance = null;
        Matrix x;
        
        instance = new SingularValueDecomposition(A.clone());
        x = instance.solve(B);
        assertTrue(A.multiply(x).equals(B, delta));

        instance = new SingularValueDecomposition(A.clone());
        x = instance.solve(C);
        assertTrue(A.multiply(x).equals(C, delta));

        instance = new SingularValueDecomposition(A.clone());
        x = instance.solve(D);
        assertTrue(A.multiply(x).equals(D, delta));

        instance = new SingularValueDecomposition(C.clone());
        x = instance.solve(E.transpose());
        assertTrue(C.multiply(x).equals(E.transpose(), delta));

        instance = new SingularValueDecomposition(C.transpose());
        x = instance.solve(E);
        assertTrue(C.transposeMultiply(x).equals(E, instance.getCondition()));
    }

    /**
     * Test of solve method, of class SingularValueDecomposition.
     */
    @Test
    public void testSolve_Matrix_ExecutorService()
    {
        System.out.println("solve");
        SingularValueDecomposition instance = null;
        Matrix x;
        
        instance = new SingularValueDecomposition(A.clone());
        x = instance.solve(B, threadpool);
        assertTrue(A.multiply(x).equals(B, delta));
        
        instance = new SingularValueDecomposition(A.clone());
        x = instance.solve(C, threadpool);
        assertTrue(A.multiply(x).equals(C, delta));
        
        instance = new SingularValueDecomposition(A.clone());
        x = instance.solve(D, threadpool);
        assertTrue(A.multiply(x).equals(D, delta));

        instance = new SingularValueDecomposition(C.clone());
        x = instance.solve(E.transpose(), threadpool);
        assertTrue(C.multiply(x).equals(E.transpose(), delta));

        instance = new SingularValueDecomposition(C.transpose());
        x = instance.solve(E, threadpool);
        assertTrue(C.transposeMultiply(x).equals(E, instance.getCondition()));
    }
}
