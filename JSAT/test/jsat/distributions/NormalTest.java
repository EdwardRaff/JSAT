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
public class NormalTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    public NormalTest()
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
     * Test of setMean method, of class Normal.
     */
    @Test
    public void testSetMean()
    {
        System.out.println("setMean");
        double mean = 0.0;
        Normal instance = new Normal();
        instance.setMean(mean);
        
        try
        {
            instance.setMean(Double.POSITIVE_INFINITY);
            fail("Can not set mean to infinity");
        }
        catch(Exception ex)
        {
            
        }
    }

    /**
     * Test of setStndDev method, of class Normal.
     */
    @Test
    public void testSetStndDev()
    {
        System.out.println("setStndDev");
        Normal instance = new Normal();
        instance.setStndDev(1.0);
        try
        {
            instance.setStndDev(Double.POSITIVE_INFINITY);
            fail("Can not set stnd deviation to infinity");
        }
        catch(Exception ex)
        {
            
        }
        
        try
        {
            instance.setStndDev(0);
            fail("Can not set stnd deviation to 0");
        }
        catch(Exception ex)
        {
            
        }
    }

    /**
     * Test of cdf method, of class Normal.
     */
    @Test
    public void testCdf_3args()
    {
        System.out.println("cdf");
        double[] cdfMean0Stnd1 = new double[]
        {
            0.0013498980316301, 0.00297976323505456, 0.00620966532577614, 0.0122244726550447,
            0.0227501319481792, 0.0400591568638171, 0.0668072012688581, 0.105649773666855, 
            0.158655253931457, 0.226627352376868, 0.308537538725987, 0.401293674317076, 0.5,
            0.598706325682924, 0.691462461274013, 0.773372647623132, 0.841344746068543, 
            0.894350226333145, 0.933192798731142, 0.959940843136183, 0.977249868051821, 
            0.987775527344955, 0.993790334674224, 0.997020236764945, 0.99865010196837
        };
        double[] cdfMean1p5Stnd2 = new double[]
        {
            0.0122244726550447, 0.0167933064484488, 0.0227501319481792, 0.0303963617652614,
            0.0400591568638171, 0.0520812794152196, 0.0668072012688581, 0.0845657223513358,
            0.105649773666855, 0.130294517136809, 0.158655253931457, 0.190786952852511, 
            0.226627352376868, 0.265985529048701, 0.308537538725987, 0.353830233327276, 
            0.401293674317076, 0.450261775169887, 0.5, 0.549738224830113, 0.598706325682924,
            0.646169766672724, 0.691462461274013, 0.7340144709513, 0.773372647623132
        };
        
        for(int i = 0; i < range.length; i++)
            assertEquals(cdfMean0Stnd1[i], Normal.cdf(range[i], 0, 1), 1e-10);
        for(int i = 0; i < range.length; i++)
            assertEquals(cdfMean1p5Stnd2[i], Normal.cdf(range[i], 1.5, 2), 1e-10);
    }

    /**
     * Test of cdf method, of class Normal.
     */
    @Test
    public void testCdf_double()
    {
        System.out.println("cdf");
        double[] cdfMean0Stnd1 = new double[]
        {
            0.0013498980316301, 0.00297976323505456, 0.00620966532577614, 0.0122244726550447,
            0.0227501319481792, 0.0400591568638171, 0.0668072012688581, 0.105649773666855, 
            0.158655253931457, 0.226627352376868, 0.308537538725987, 0.401293674317076, 0.5,
            0.598706325682924, 0.691462461274013, 0.773372647623132, 0.841344746068543, 
            0.894350226333145, 0.933192798731142, 0.959940843136183, 0.977249868051821, 
            0.987775527344955, 0.993790334674224, 0.997020236764945, 0.99865010196837
        };
        double[] cdfMean1p5Stnd2 = new double[]
        {
            0.0122244726550447, 0.0167933064484488, 0.0227501319481792, 0.0303963617652614,
            0.0400591568638171, 0.0520812794152196, 0.0668072012688581, 0.0845657223513358,
            0.105649773666855, 0.130294517136809, 0.158655253931457, 0.190786952852511, 
            0.226627352376868, 0.265985529048701, 0.308537538725987, 0.353830233327276, 
            0.401293674317076, 0.450261775169887, 0.5, 0.549738224830113, 0.598706325682924,
            0.646169766672724, 0.691462461274013, 0.7340144709513, 0.773372647623132
        };
        
        Normal dist = new Normal(0, 1);
        for(int i = 0; i < range.length; i++)
            assertEquals(cdfMean0Stnd1[i], dist.cdf(range[i]), 1e-10);
        dist = new Normal(1.5, 2);
        for(int i = 0; i < range.length; i++)
            assertEquals(cdfMean1p5Stnd2[i], dist.cdf(range[i]), 1e-10);
        
                
    }

    /**
     * Test of invcdf method, of class Normal.
     */
    @Test
    public void testInvcdf()
    {
        System.out.println("invcdf");
        double[] inCDF = new double[]
        {
            -3.3000727542547823, -1.8057072650340151, -1.1794924187821967, -0.7419660787053934, -0.39155417237863954, 
            -0.09132059856786046, 0.17670363698649716, 0.4228030336829398, 0.6535501073566241, 0.8735200785359454, 
            1.0861426807957881, 1.2941759325571114, 1.5, 1.7058240674428888, 1.9138573192042119, 2.1264799214640546, 
            2.3464498926433754, 2.57719696631706, 2.823296363013503, 3.09132059856786, 3.3915541723786395, 
            3.741966078705394, 4.179492418782196, 4.805707265034015, 6.300072754254788
        };
        
        for(int i = 0; i < range.length; i++)
            assertEquals(inCDF[i], Normal.invcdf(range[i]/6.1+0.5, 1.5, 2), 1e-10);
    }

    /**
     * Test of invCdf method, of class Normal.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        Normal instance = new Normal(1.5, 2);
        double[] inCDF = new double[]
        {
            -3.3000727542547823, -1.8057072650340151, -1.1794924187821967, -0.7419660787053934, -0.39155417237863954, 
            -0.09132059856786046, 0.17670363698649716, 0.4228030336829398, 0.6535501073566241, 0.8735200785359454, 
            1.0861426807957881, 1.2941759325571114, 1.5, 1.7058240674428888, 1.9138573192042119, 2.1264799214640546, 
            2.3464498926433754, 2.57719696631706, 2.823296363013503, 3.09132059856786, 3.3915541723786395, 
            3.741966078705394, 4.179492418782196, 4.805707265034015, 6.300072754254788
        };
        
        for(int i = 0; i < range.length; i++)
            assertEquals(inCDF[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of pdf method, of class Normal.
     */
    @Test
    public void testPdf_3args()
    {
        System.out.println("pdf");
        double[] pdfMean0Stnd1 = new double[]
        {
            0.00443184841193801, 0.00909356250159105, 0.0175283004935685, 0.0317396518356674, 0.0539909665131881, 
            0.0862773188265115, 0.129517595665892, 0.182649085389022, 0.241970724519143, 0.301137432154804, 
            0.3520653267643, 0.386668116802849, 0.398942280401433, 0.386668116802849, 0.3520653267643, 
            0.301137432154804, 0.241970724519143, 0.182649085389022, 0.129517595665892, 0.0862773188265115, 
            0.0539909665131881, 0.0317396518356674, 0.0175283004935685, 0.00909356250159105, 0.00443184841193801
        };
        double[] pdfMean1p5Stnd2 = new double[]
        {
            0.0158698259178337, 0.0208604926281693, 0.026995483256594, 0.034393137913346, 0.0431386594132558, 
            0.0532691340652925, 0.0647587978329459, 0.0775061327291466, 0.091324542694511, 0.10593832288785, 
            0.120985362259572, 0.136027499189272, 0.150568716077402, 0.164080484275188, 0.17603266338215, 
            0.185927546934885, 0.193334058401425, 0.197918843472375, 0.199471140200716, 0.197918843472375, 
            0.193334058401425, 0.185927546934885, 0.17603266338215, 0.164080484275188, 0.150568716077402
        };
        
        for(int i = 0; i < range.length; i++)
            assertEquals(pdfMean0Stnd1[i], Normal.pdf(range[i], 0, 1), 1e-10);
        for(int i = 0; i < range.length; i++)
            assertEquals(pdfMean1p5Stnd2[i], Normal.pdf(range[i], 1.5, 2), 1e-10);
    }

    /**
     * Test of pdf method, of class Normal.
     */
    @Test
    public void testPdf_double()
    {
        System.out.println("pdf");
        double[] pdfMean0Stnd1 = new double[]
        {
            0.00443184841193801, 0.00909356250159105, 0.0175283004935685, 0.0317396518356674, 0.0539909665131881, 
            0.0862773188265115, 0.129517595665892, 0.182649085389022, 0.241970724519143, 0.301137432154804, 
            0.3520653267643, 0.386668116802849, 0.398942280401433, 0.386668116802849, 0.3520653267643, 
            0.301137432154804, 0.241970724519143, 0.182649085389022, 0.129517595665892, 0.0862773188265115, 
            0.0539909665131881, 0.0317396518356674, 0.0175283004935685, 0.00909356250159105, 0.00443184841193801
        };
        double[] pdfMean1p5Stnd2 = new double[]
        {
            0.0158698259178337, 0.0208604926281693, 0.026995483256594, 0.034393137913346, 0.0431386594132558, 
            0.0532691340652925, 0.0647587978329459, 0.0775061327291466, 0.091324542694511, 0.10593832288785, 
            0.120985362259572, 0.136027499189272, 0.150568716077402, 0.164080484275188, 0.17603266338215, 
            0.185927546934885, 0.193334058401425, 0.197918843472375, 0.199471140200716, 0.197918843472375, 
            0.193334058401425, 0.185927546934885, 0.17603266338215, 0.164080484275188, 0.150568716077402
        };
        
        Normal dist = new Normal(0, 1);
        for(int i = 0; i < range.length; i++)
            assertEquals(pdfMean0Stnd1[i], dist.pdf(range[i]), 1e-10);
        dist = new Normal(1.5, 2);
        for(int i = 0; i < range.length; i++)
            assertEquals(pdfMean1p5Stnd2[i], dist.pdf(range[i]), 1e-10);
    }
    
    @Test
    public void testLogPDF_double()
    {
        double[] logPdfMean0Std1 = new double[]
        {
            -5.4189385332046727418, -4.7001885332046727418, -4.0439385332046727418, 
            -3.4501885332046727418, -2.9189385332046727418, -2.4501885332046727418, 
            -2.0439385332046727418, -1.7001885332046727418, -1.4189385332046727418, 
            -1.2001885332046727418, -1.0439385332046727418, -0.95018853320467274178, 
            -0.91893853320467274178, -0.95018853320467274178, -1.0439385332046727418, 
            -1.2001885332046727418, -1.4189385332046727418, -1.7001885332046727418,
            -2.0439385332046727418, -2.4501885332046727418, -2.9189385332046727418,
            -3.4501885332046727418, -4.0439385332046727418, -4.7001885332046727418,
            -5.4189385332046727418
        };
        
        double[] logPdfMean1p5Std2 = new double[]
        {
            -4.1433357137646180512, -3.8698982137646180512, -3.6120857137646180512, 
            -3.3698982137646180512, -3.1433357137646180512, -2.9323982137646180512,
            -2.7370857137646180512, -2.5573982137646180512, -2.3933357137646180512,
            -2.2448982137646180512, -2.1120857137646180512, -1.9948982137646180512,
            -1.8933357137646180512, -1.8073982137646180512, -1.7370857137646180512,
            -1.6823982137646180512, -1.6433357137646180512, -1.6198982137646180512,
            -1.6120857137646180512, -1.6198982137646180512, -1.6433357137646180512, 
            -1.6823982137646180512, -1.7370857137646180512, -1.8073982137646180512,
            -1.8933357137646180512
        };
        
        Normal dist = new Normal(0, 1);
        for(int i = 0; i < range.length; i++)
            assertEquals(logPdfMean0Std1[i], dist.logPdf(range[i]), 1e-8);
        dist = new Normal(1.5, 2);
        for(int i = 0; i < range.length; i++)
            assertEquals(logPdfMean1p5Std2[i], dist.logPdf(range[i]), 1e-8);
    }


    /**
     * Test of min method, of class Normal.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        Normal instance = new Normal();
        assertTrue(instance.min() == Double.NEGATIVE_INFINITY);
    }

    /**
     * Test of max method, of class Normal.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        Normal instance = new Normal();
        assertTrue(instance.max() == Double.POSITIVE_INFINITY);
    }

    /**
     * Test of mean method, of class Normal.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        Normal instance = new Normal(0, 1);
        assertEquals(0.0, instance.mean(), 1e-10);
    }

    /**
     * Test of median method, of class Normal.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        Normal instance = new Normal(0, 1);
        assertEquals(0.0, instance.median(), 1e-10);
    }

    /**
     * Test of mode method, of class Normal.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        Normal instance = new Normal(0, 1);
        assertEquals(0.0, instance.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class Normal.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        Normal instance = new Normal(0, 1);
        assertEquals(1, instance.variance(), 1e-10);
        
        instance = new Normal(0, 2);
        assertEquals(4, instance.variance(), 1e-10);
    }

    /**
     * Test of standardDeviation method, of class Normal.
     */
    @Test
    public void testStandardDeviation()
    {
        System.out.println("standardDeviation");
        Normal instance = new Normal(0, 1);
        assertEquals(1, instance.standardDeviation(), 1e-10);
        
        instance = new Normal(0, 2);
        assertEquals(2, instance.standardDeviation(), 1e-10);
    }

    /**
     * Test of skewness method, of class Normal.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        Normal instance = new Normal(0, 1);
        assertEquals(0, instance.skewness(), 1e-10);
        
        instance = new Normal(0, 2);
        assertEquals(0, instance.skewness(), 1e-10);
    }
    @Test
    public void testEquals(){
    	System.out.println("equals");
    	ContinuousDistribution d1 = new Normal(0.5, 0.5);
    	ContinuousDistribution d2 = new Normal(0.6, 0.5);
    	ContinuousDistribution d3 = new Normal(0.5, 0.6);
    	ContinuousDistribution d4 = new Normal(0.5, 0.5);
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
    	ContinuousDistribution d1 = new Normal(0.5, 0.5);
    	ContinuousDistribution d2 = new Normal(0.6, 0.5);
    	ContinuousDistribution d4 = new Normal(0.5, 0.5);
    	assertEquals(d1.hashCode(), d4.hashCode());
    	assertFalse(d1.hashCode()==d2.hashCode());
    }
    
    @Test
    public void testHugeRange()
    {
        Normal n = new Normal(811.4250871080139d, 1540.8594859716793d);
        //original tests from TKlerx that failed 
        assertTrue(n.cdf(44430.0d) <= 1);
        assertFalse(Double.isNaN(n.cdf(67043.0)));
        
        //Some more tests 
        assertTrue(n.cdf(-44430.0d) >= 0);
        assertFalse(Double.isNaN(n.cdf(-67043.0)));
        
        for(double v : new double[]{44430.0d, -44430.0d, 67043, -67043})
        assertTrue(n.pdf(v) >= 0 && n.pdf(v) <= 1e-20);
    }
}
