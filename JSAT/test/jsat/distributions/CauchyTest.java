/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.distributions;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class CauchyTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public CauchyTest()
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
    }

    

    /**
     * Test of pdf method, of class Cauchy.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0.012732395447351628,0.014719532309077025,0.017205939793718417,0.020371832715762605,0.02448537586029159,0.0299585775231803,0.03744822190397538,0.048046775273025005,0.06366197723675814,0.08780962377483881,0.12732395447351627,0.19588300688233273,0.3183098861837907,0.5092958178940651,0.6366197723675814,0.5092958178940651,0.3183098861837907,0.19588300688233273,0.12732395447351627,0.08780962377483881,0.06366197723675814,0.048046775273025005,0.03744822190397538,0.0299585775231803,0.02448537586029159};
        double[] paramTwo1 = new double[]{0.04493786628477045,0.048814295644798576,0.05305164769729845,0.057656130327630006,0.06261833826566374,0.067906109052542,0.07345612758087477,0.07916515304052826,0.08488263631567752,0.0904075416379997,0.09549296585513718,0.09986192507726767,0.1032356387623105,0.10537154852980657,0.1061032953945969,0.10537154852980657,0.1032356387623105,0.09986192507726767,0.09549296585513718,0.0904075416379997,0.08488263631567752,0.07916515304052826,0.07345612758087477,0.067906109052542,0.06261833826566374};
        double[] paramTwo2 = new double[]{0.00439048118874194,0.0047776343142032374,0.005218194855471979,0.005722424920158035,0.00630316606304536,0.006976655039644728,0.007763655760580261,0.008691054912868005,0.009794150344116638,0.011119996023887883,0.012732395447351628,0.014719532309077025,0.017205939793718417,0.020371832715762605,0.02448537586029159,0.0299585775231803,0.03744822190397538,0.048046775273025005,0.06366197723675814,0.08780962377483881,0.12732395447351627,0.19588300688233273,0.3183098861837907,0.5092958178940651,0.6366197723675814};
        double[] paramTwo3 = new double[]{0.02122065907891938,0.02270263675605045,0.02432941805226426,0.026117734250977697,0.028086166427981528,0.030255197102617728,0.03264716781372212,0.035286084380651166,0.03819718634205488,0.04140616405642805,0.04493786628477045,0.048814295644798576,0.05305164769729845,0.057656130327630006,0.06261833826566374,0.067906109052542,0.07345612758087477,0.07916515304052826,0.08488263631567752,0.0904075416379997,0.09549296585513718,0.09986192507726767,0.1032356387623105,0.10537154852980657,0.1061032953945969};
        
        instance = new Cauchy(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.pdf(range[i]), 1e-10);
        instance = new Cauchy(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.pdf(range[i]), 1e-10);
        instance = new Cauchy(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.pdf(range[i]), 1e-10);
        instance = new Cauchy(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.pdf(range[i]), 1e-10);
    }

    /**
     * Test of cdf method, of class Cauchy.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0.045167235300866526,0.04858979034752886,0.052568456711253375,0.0572491470487001,0.06283295818900114,0.06960448727306395,0.07797913037736925,0.08858553278290471,0.10241638234956668,0.12111894159084341,0.14758361765043326,0.1871670418109988,0.25,0.35241638234956674,0.5,0.6475836176504333,0.75,0.8128329581890013,0.8524163823495667,0.8788810584091566,0.8975836176504333,0.9114144672170953,0.9220208696226307,0.9303955127269361,0.9371670418109989};
        double[] paramTwo1 = new double[]{0.22556274802780257,0.23727438865200817,0.25,0.2638308495666619,0.27885793837630446,0.2951672353008665,0.3128329581890012,0.3319086824248374,0.35241638234956674,0.3743340836219976,0.39758361765043326,0.4220208696226307,0.44743154328874657,0.47353532394041015,0.5,0.5264646760595899,0.5525684567112534,0.5779791303773694,0.6024163823495667,0.6256659163780024,0.6475836176504333,0.6680913175751626,0.6871670418109987,0.7048327646991335,0.7211420616236955};
        double[] paramTwo2 = new double[]{0.026464676059589853,0.027609670711723877,0.02885793837630446,0.030224066838919483,0.03172551743055352,0.033383366430525085,0.035223287477277265,0.03727687115420514,0.03958342416056554,0.04219246315884134,0.045167235300866526,0.04858979034752886,0.052568456711253375,0.0572491470487001,0.06283295818900114,0.06960448727306395,0.07797913037736925,0.08858553278290471,0.10241638234956668,0.12111894159084341,0.14758361765043326,0.1871670418109988,0.25,0.35241638234956674,0.5};
        double[] paramTwo3 = new double[]{0.14758361765043326,0.15307117542621002,0.15894699814425117,0.16524934053856788,0.17202086962263063,0.17930913508098678,0.1871670418109988,0.1956532942677373,0.20483276469913347,0.21477671252272273,0.22556274802780257,0.23727438865200817,0.25,0.2638308495666619,0.27885793837630446,0.2951672353008665,0.3128329581890012,0.3319086824248374,0.35241638234956674,0.3743340836219976,0.39758361765043326,0.4220208696226307,0.44743154328874657,0.47353532394041015,0.5};
        
        instance = new Cauchy(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.cdf(range[i]), 1e-10);
        instance = new Cauchy(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.cdf(range[i]), 1e-10);
        instance = new Cauchy(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.cdf(range[i]), 1e-10);
        instance = new Cauchy(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.cdf(range[i]), 1e-10);
    }

    /**
     * Test of invCdf method, of class Cauchy.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{-18.912611074230398,-2.7103586757986284,-1.2177088085070724,-0.644097849410133,-0.3326793086091363,-0.13173366628867933,0.012712405159932383,0.12479067856711157,0.2170191767409292,0.29665435282523944,0.3683227410588536,0.43526498965126525,0.5,0.5647350103487349,0.6316772589411463,0.7033456471747606,0.7829808232590707,0.8752093214328884,0.9872875948400677,1.131733666288679,1.3326793086091362,1.644097849410134,2.2177088085070724,3.7103586757986284,19.912611074230565};
        double[] paramTwo1 = new double[]{-115.97566644538239,-18.76215205479177,-9.806252851042434,-6.364587096460799,-4.496075851654818,-3.2904019977320758,-2.4237255690404056,-1.7512559285973306,-1.1978849395544249,-0.7200738830485633,-0.2900635536468784,0.11158993790759136,0.5,0.8884100620924091,1.2900635536468785,1.7200738830485633,2.1978849395544247,2.7512559285973306,3.4237255690404056,4.290401997732075,5.496075851654818,7.364587096460804,10.806252851042434,19.76215205479177,116.97566644538338};
        double[] paramTwo2 = new double[]{-16.412611074230398,-0.21035867579862844,1.2822911914929276,1.855902150589867,2.167320691390864,2.3682663337113206,2.5127124051599323,2.624790678567112,2.717019176740929,2.7966543528252394,2.8683227410588534,2.935264989651265,3.,3.064735010348735,3.1316772589411466,3.2033456471747606,3.282980823259071,3.375209321432888,3.4872875948400677,3.631733666288679,3.832679308609136,4.144097849410134,4.717708808507073,6.210358675798629,22.412611074230565};
        double[] paramTwo3 = new double[]{-113.47566644538239,-16.26215205479177,-7.306252851042434,-3.8645870964607987,-1.996075851654818,-0.7904019977320758,0.07627443095959441,0.7487440714026694,1.3021150604455751,1.7799261169514367,2.2099364463531215,2.6115899379075915,3.,3.388410062092409,3.7900635536468785,4.220073883048563,4.697884939554425,5.251255928597331,5.923725569040405,6.790401997732075,7.996075851654818,9.864587096460804,13.306252851042434,22.26215205479177,119.47566644538338};
        
        instance = new Cauchy(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Cauchy(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Cauchy(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Cauchy(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of min method, of class Cauchy.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        Cauchy instance = new Cauchy();
        assertTrue(Double.NEGATIVE_INFINITY == instance.min());
    }

    /**
     * Test of max method, of class Cauchy.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        Cauchy instance = new Cauchy();
        assertTrue(Double.POSITIVE_INFINITY == instance.max());
    }

    /**
     * Test of mean method, of class Cauchy.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        Cauchy instance = new Cauchy();
        assertTrue(Double.isNaN(instance.mean()));
    }

    /**
     * Test of median method, of class Cauchy.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new Cauchy(0.5, 0.5);
        assertEquals(0.5, dist.median(), 1e-10);
        dist = new Cauchy(0.5, 3);
        assertEquals(0.5, dist.median(), 1e-10);
        dist = new Cauchy(3, 0.5);
        assertEquals(3, dist.median(), 1e-10);
        dist = new Cauchy(3, 3);
        assertEquals(3, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class Cauchy.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new Cauchy(0.5, 0.5);
        assertEquals(0.5, dist.mode(), 1e-10);
        dist = new Cauchy(0.5, 3);
        assertEquals(0.5, dist.mode(), 1e-10);
        dist = new Cauchy(3, 0.5);
        assertEquals(3, dist.mode(), 1e-10);
        dist = new Cauchy(3, 3);
        assertEquals(3, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class Cauchy.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        Cauchy instance = new Cauchy();
        assertTrue(Double.isNaN(instance.variance()));
    }

    /**
     * Test of standardDeviation method, of class Cauchy.
     */
    @Test
    public void testStandardDeviation()
    {
        System.out.println("standardDeviation");
        Cauchy instance = new Cauchy();
        assertTrue(Double.isNaN(instance.standardDeviation()));
    }

    /**
     * Test of skewness method, of class Cauchy.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        Cauchy instance = new Cauchy();
        assertTrue(Double.isNaN(instance.skewness()));
    }
    @Test
    public void testEquals(){
    	System.out.println("equals");
    	ContinuousDistribution d1 = new Cauchy(0.5, 0.5);
    	ContinuousDistribution d2 = new Cauchy(0.6, 0.5);
    	ContinuousDistribution d3 = new Cauchy(0.5, 0.6);
    	ContinuousDistribution d4 = new Cauchy(0.5, 0.5);
    	Integer i = new Integer(1);
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
    public void testHashCode(){
    	System.out.println("hashCode");
    	ContinuousDistribution d1 = new Cauchy(0.5, 0.5);
    	ContinuousDistribution d2 = new Cauchy(0.6, 0.5);
    	ContinuousDistribution d4 = new Cauchy(0.5, 0.5);
    	assertEquals(d1.hashCode(), d4.hashCode());
    	assertFalse(d1.hashCode()==d2.hashCode());
    }
}
